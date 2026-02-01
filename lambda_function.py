import json
import time
import boto3
import os
import re

s3_client = boto3.client('s3')
bedrock_client = boto3.client("bedrock-runtime")
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
                    "category": skill["skill_category"],
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

def invoke_bedrock(system_prompt, messages):
    response = bedrock_client.converse(
        modelId="amazon.nova-micro-v1:0",
        messages=messages,
        system=system_prompt,
        inferenceConfig={
            "maxTokens": 2000,
            "temperature": 0.4
        }
    )
    print("Bedrock response: ", response)
    return response["output"]["message"]["content"][0]["text"]

def add_future_pathways(skills_of_interest):
    system_prompt = [{
        "text": '''
            In about 20 words, list a few ways that graduating high school student could further their
            development of a given skill. Include potential courses of study, professional certifications,
            or careers that value and develop that skill. Write in full sentences in imperative form.
            Include at least two clauses.
        '''
    }]
    
    for skill in skills_of_interest:
        user_messages = [{
            "role": "user",
            "content": [{
                "text": skill["name"]
            }]
        }]
        skill["pathways"] = invoke_bedrock(system_prompt, user_messages)

def llm_summary(skills_of_interest):
    system_prompt = [{
        "text": '''
            Write a summary to go at the end of a transcript skill analysis for a high school student.
            The primary goal of the summary should be to reinforce to the student that their transcript represents
            real skills that are useful and can help them reach their goals. The blurb should be less than 100 words,
            positive in tone, and written in the second person. Don't be too sycophantic or make specific assertions
            about what they are qualified to do. For example, given the skills Writing, Critical Thinking /
            Problem Solving, Woodworking, Research, and Problem Solving a good summary might look like: 
            "You can write clearly, question assumptions, and finish a woodworking project without splinters. 
            Research papers don't intimidate you, and you've learned to map big assignments into small, doable steps.
            These skills will carry you well into your future and beyond."            
        '''
    }]
    user_messages = [{
        "role": "user",
        "content": [{
            "text": ", ".join([skill["name"] for skill in skills_of_interest])
        }]
    }]
    
    return invoke_bedrock(system_prompt, user_messages)

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
    full_skills_of_interest = [student_skills[id] for id in skills_of_interest]
    add_future_pathways(full_skills_of_interest)
    
    print("Highest count skill:", full_skills_of_interest[0])
    print("Highest level skill:", full_skills_of_interest[1])
    print("Most unique skill:", full_skills_of_interest[2])
    
    summary = llm_summary(full_skills_of_interest)
    
    analyzed_course_ids = [course["code"] for course in course_skills_data]
    response = {
        'status': 200,
        'body': {
            "count": str(len(student_skills)),
            "skills_of_interest": full_skills_of_interest,
            "summary": summary,
            "course_ids": analyzed_course_ids,
        }
    }
    return response
