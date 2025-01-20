search_query_prompt = """ 
    You are an expert at researching people online. Your goal is to find detailed information about a candidate for a job opportunity.
    The candidate is:
    {candidate_full_name}
    {candidate_context}
    The job they're being considered for is:
    {job_description}
    Generate {number_of_queries} search queries that will help gather comprehensive information about this candidate. 
    
    IMPORTANT: Each query should:
    - Be 2-5 words maximum
    - Include the person's full name
    - Focus on ONE specific aspect (hometown, school, company, etc.)
    - Be something you would actually type into Google

    Example good queries for "John Smith":
    - "John Smith Microsoft"
    - "John Smith Hinsdale Illinois"
    - "John Smith Stanford"
    - "John Smith GitHub"

    Example bad queries (too complex):
    - "John Smith software engineer Microsoft technical projects"
    - "John Smith professional background experience skills"
    - "John Smith publications and conference presentations"

    Focus on queries that might find:
    - Professional profiles (Portfolios, GitHub, Google Scholar, etc - whatever is relevant to the job)
    - Company mentions
    - University connections
    - Projects, publications, etc.
"""


validation_prompt = """
    You are a validator determining if a webpage's content is genuinely about a specific candidate.

    Candidate Full Name: {candidate_full_name}
    Candidate Profile:
    {candidate_context}
    Raw Content: {raw_content}

    Use the following guidelines to validate if this webpage is about the candidate in question:
    1. **Name Match**:
    - The webpage must explicitly mention the candidate's full name or a clear variation.

    2. **Context Alignment**:
    - Current or past employers mentioned in the candidate's profile.
    - Educational institutions from the candidate's background.
    - Job titles or roles from the candidate's experience.
    - Projects or achievements mentioned in the candidate's profile.
    - Time periods that align with the candidate's career history.

    3. **Confidence Check**:
    - Is there any conflicting information that suggests this might be about a different person?
    - Are there enough specific details to be confident this is about our candidate?
    - Could this content reasonably apply to someone else with the same name?

    While you should be very careful in your evaluation, we don't want to reject a valid source. Provide a confidence score between `0` and `1`, with anything above `0.8` being a valid source.
"""


distill_source_prompt = """
    You will be given a string of raw content from a webpage.
    Please extract the relevant information about the given person from the raw HTML.
    Describe what the source is, what it is about, and how it is relevant to the person, etc.
    Write your response in paragraph form.

    Limit the response to 150 words.

    Here is the raw content:
    {raw_content}

    Here is the person's full name:
    {candidate_full_name}
"""
