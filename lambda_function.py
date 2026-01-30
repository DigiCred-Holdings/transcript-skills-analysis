import json
import time
import boto3
import os
import re

s3_client = boto3.client('s3')
REGISTRY_URI = os.environ['REGISTRY_S3_URI']

def load_skills_dataset():
    # Get bucket key from environment variable S3 URI e.g. s3://digicred-credential-analysis/dev/staging_registry.json
    bucket, key = REGISTRY_URI.replace("s3://", "").split("/", 1)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    if response['ResponseMetadata']['HTTPStatusCode'] != 200:
        raise Exception(f"Failed to retrieve data from S3: {response['ResponseMetadata']['HTTPStatusCode']}")
    try:
        content = response['Body'].read().decode('utf-8')
        result = json.loads(content)
    except Exception as e:
        raise Exception(f'Failed to parse data from s3:', e)
    return result

def find_relevant_courses(course_title_code_list, all_courses):
    all_course_codes = [course["code"].upper() for course in all_courses if course["code"]]
    found_student_courses = []
    missing_codes = []
    for given_title, given_code in course_title_code_list:
        candidates = []
        for code_to_evaluate in all_course_codes:
            if given_code in code_to_evaluate:
                candidates += [course for course in all_courses if course.get("code") == code_to_evaluate]

        if len(candidates) == 1:
            found_student_courses.append(candidates[0])
        elif len(candidates):
            print(f"{len(candidates)} candidates for course were found in the registry for code", given_code, given_title)
            print(", ".join([course["code"] + ": " + course["name"] for course in candidates]))
            missing_codes.append([given_title, given_code])
        else:
            print(f"Course code was not found in the registry", given_code)
            missing_codes.append([given_title, given_code])

    print(f"Warning: {len(missing_codes)} courses were not found in the database.")
    print(f"Could not find the following courses in registry: {missing_codes}")
    return found_student_courses


def get_course_data(course_title_code_list):
    all_courses = load_skills_dataset()
    course_skill_data = find_relevant_courses(course_title_code_list, all_courses)    
    return course_skill_data

def get_highest_count_skill(course_skill_data):
    skill_id_count = {}
    for course in course_skill_data:
        for skill in course["skills_curated"]:
            id = skill["skill_id"]
            if id not in skill_id_count:
                skill_id_count[id] = {
                    "count": 1,
                    "skill": skill,
                    "course_code": [course["code"]]
                }
            else:
                skill_id_count[id]["count"] += 1
                skill_id_count[id]["course_code"].append(course["code"])
    
    print(skill_id_count)
    max_count_skill = None
    max_count = 0
    for skill_id, skill_data in skill_id_count.items():
        if skill_data["count"] > max_count:
            max_count_skill = skill_data
            max_count = skill_data["count"]

    return max_count_skill
        

def invoke_bedrock_model(messages: list[dict[str, str]]):
    client = boto3.client("bedrock-runtime")

    # Build the conversation for the Converse API
    system_prompt = []
    conversation = []
    for msg in messages:
        role = msg["role"]  # 'system'|'user'|'assistant'
        if role == "system":
            system_prompt.append({"text": msg["content"]})
            continue
        content = [{"text": msg["content"]}]
        conversation.append({"role": role, "content": content})

    response = client.converse(
        modelId="amazon.nova-micro-v1:0",
        messages=conversation,
        system=system_prompt,
        inferenceConfig={
            "maxTokens": 2000,
            "temperature": 0.0
        }
    )

    assistant_msg = response["output"]["message"]["content"][0]["text"]
    return assistant_msg


def get_prompt(course_skills_data):
    course_descriptions = [(course["title"], course["description"]) for course in course_skills_data]
    skills_by_course = [(course["title"], course["skills"]) for course in course_skills_data]
    prompt = [
        {"role": "system", "content": '''
            You are summarizing a university-level student's abilities and skills.
            You will receive:
            1) A list of completed courses with descriptions
            2) A list of skills associated with those courses

            Your task:
            - Write a short summary (max 3 sentences) of the student's strengths.
            - Mention at least one notable skill group they excel in.
            - Highlight at least one specific skill learned in a course (referencing course context).
            - Keep the tone positive, in the style of: "Your coursework has given you skills in ... Notably your accounting class taught you ..."
            - Avoid lists; keep it narrative and concise.

            Output only the 3-sentence summary.
        '''},
        {"role": "user", "content": f'''
            1) {course_descriptions}
            2) {skills_by_course}
        '''}
    ]

    return prompt


def chatgpt_summary(course_skills_data):
    prompt = get_prompt(course_skills_data)
    summary = invoke_bedrock_model(prompt)
    return summary


def compile_highlight(summary, course_skills_data):

    # Helper to clean skill strings of leading numbers/formatting
    def clean_skill(s):
        return re.sub(r"^\s*[\d]+[.)]\s*", "", str(s)).strip()

    # Build a list of standout skills by selecting the most common skills across courses
    skill_counts = {}
    for course in course_skills_data:
        skills = [clean_skill(s) for s in course["skills"]]
        for skill in skills:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1

    # Pick the top 3 most common skills as standout skills
    sorted_skills = sorted(skill_counts.items(), key=lambda item: item[1], reverse=True)
    top_3_skills = [s[0] for s in sorted_skills[:3]]
    
    # Format standout list 
    standout_sentence = ""
    quoted = [f"'{s}'" for s in top_3_skills]
    if len(quoted) > 1:
        quoted_str = ", ".join(quoted[:-1]) + f", and {quoted[-1]}"
    else:
        quoted_str = quoted[0]
    standout_sentence = f" Some of your standout skills are {quoted_str}."

    # Final 'totals' sentence
    totals_sentence = ""
    totals_sentence = (
        f"Overall, we have analyzed {len(course_skills_data)} of your courses "
        f"and found {len(skill_counts.keys())} skills!"
    )
    totals_sentence += standout_sentence

    # Build the highlight
    highlight = f"{summary}\n\n{totals_sentence}".strip()
    return highlight


from time import perf_counter
def _timeit(f):
    def wrap(*a, **kw):
        t=perf_counter(); r=f(*a, **kw)
        print(f"{f.__name__} took {(perf_counter()-t)*1000:.3f} ms")
        return r
    return wrap


@_timeit
def lambda_handler(event, context):
    if type(event["body"]) is str:
        body = json.loads(event["body"])
    else:
        body = event["body"]
    if not body:
        return {
            'statusCode': 400,
            'body': 'Invalid input: body cannot be empty.'
        }
    
    ()
    if "coursesList" not in body:
        return {
            'statusCode': 400,
            'body': 'Invalid input: coursesList and source are required.'
        }
        

    course_skills_data = get_course_data(body["coursesList"])
    highest_count_skill = get_highest_count_skill(course_skills_data)
    print(f"Highest count skill: {highest_count_skill}")
    
    summary = chatgpt_summary(course_skills_data)
    
    # highlight = compile_highlight(summary, course_skills_data)
    
    analyzed_course_ids = [course["id"] for course in course_skills_data]
    response = {
        'status': 200,
        'body': {
            "summary": summary,
            "course_ids": analyzed_course_ids,
        }
    }
    return response
