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


def package_skills(course_skill_data):
    student_skills = {}
    for course in course_skill_data:
        for skill in course["skills_curated"]:
            id = skill["skill_id"]
            if id not in student_skills:
                student_skills[id] = {
                    "name": skill["skill"],
                    "category": skill["category"],
                    "frequency": skill["frequency"],
                    "count": 1,
                    "max_skill_level": skill["skill_level"],
                    "sum_skill_level": skill["skill_level"],
                    "courses": [(course["code"], skill["skill_level"])]
                }
            else:
                student_skills[id]["count"] += 1
                if skill["skill_level"] > student_skills[id]["max_skill_level"]: 
                    student_skills[id]["max_skill_level"] = skill["skill_level"]
                student_skills[id]["sum_skill_level"] += skill["skill_level"]
                student_skills[id]["courses"].append((course["code"], skill["skill_level"]))
    return student_skills

def get_skills_of_interest(all_skills):
    max_count_skill = None
    max_count = 0
    max_level_skill = None
    max_average_level = 0
    unique_skill = None
    unique_skill_frequency = float("inf")
    for id, skill_data in all_skills.items():
        if skill_data["count"] > max_count:
            max_count_skill = id
            max_count = skill_data["count"]
        
        skill_average = skill_data["sum_skill_level"] / len(skill_data["courses"])
        skill_data["skill_level_average"] = skill_average
        if skill_average > max_average_level:
            max_level_skill = id
            max_average_level = skill_average
        
        if skill_data["frequency"] < unique_skill_frequency:
            unique_skill = id
            unique_skill_frequency = skill_data["frequency"]
    

    return [max_count_skill, max_level_skill, unique_skill]


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
    student_skills = package_skills(course_skills_data)
    skills_of_interest = get_skills_of_interest(student_skills)
    
    print("Highest count skill:", skills_of_interest[0])
    print("Highest level skill:", skills_of_interest[1])
    print("Most unique skill:", skills_of_interest[2])
    
    # summary = chatgpt_summary(course_skills_data)
    
    # highlight = compile_highlight(summary, course_skills_data)
    
    analyzed_course_ids = [course["code"] for course in course_skills_data]
    response = {
        'status': 200,
        'body': {
            "skills": student_skills,
            "skills_of_interest": skills_of_interest,
            # "summary": summary,
            "course_ids": analyzed_course_ids,
        }
    }
    return response
