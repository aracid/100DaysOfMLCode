import openai

# Define your API key
api_key = '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'

# Initialize the OpenAI API
openai.api_key = api_key

# Prompt user for business decision
decision = "Visual effects and animation company actively planning to remain relevant using AI within the market"

# List of prompts based on different models or principles
prompts = {
    "Pareto Principle (80/20 Rule)": "Analyze [my business decision] using the Pareto Principle (80/20 Rule). Identify the 20% of factors that contribute to 80% of the results and focus on optimizing those factors.",
    "SWOT Analysis": "Utilize SWOT Analysis to assess [my business decision]. Evaluate the internal strengths and weaknesses, as well as external opportunities and threats, to make an informed decision.",
    "Innovator's Dilemma": "Apply the Innovator's Dilemma to evaluate [my business decision]. Consider how disruptive innovations can challenge established products or services in the market.",
    "Hindsight Bias": "Evaluate [my business decision] while considering Hindsight Bias. Reflect on how your current knowledge might influence your perception of the decision's risks and benefits.",
    "Lateral Thinking": "Use Lateral Thinking to analyze [my business decision]. Encourage unconventional and creative approaches to problem-solving to uncover new perspectives and solutions.",
    "Maslow's Hierarchy of Needs": "Analyze [my business decision] using Maslow's Hierarchy of Needs. Consider how the decision aligns with different levels of human needs, from basic to self-actualization.",
    "Bounded Rationality": "Evaluate [my business decision] considering Bounded Rationality. Recognize that decision-making is limited by cognitive resources and analyze how this might impact the quality of the decision.",
    "Holacracy": "Apply Holacracy principles to assess [my business decision]. Consider how distributing authority and decision-making across self-organizing teams might impact the implementation.",
    "Cognitive Biases": "Utilize knowledge of Cognitive Biases to analyze [my business decision]. Identify common biases that could affect your judgment and decision-making process.",
    "Resource Allocation": "Analyze [my business decision] by considering Resource Allocation. Evaluate how the decision impacts the distribution of resources such as time, money, and manpower.",
    "Expectancy Theory": "Apply Expectancy Theory to analyze [my business decision]. Evaluate how individuals' expectations of effort, performance, and outcomes might influence their motivation and decision-making.",
    "Agile Methodology": "Utilize Agile Methodology to assess [my business decision]. Break down the decision into iterative cycles, focusing on adaptability, collaboration, and continuous improvement.",
    "Stakeholder Mapping": "Analyze [my business decision] through Stakeholder Mapping. Identify and assess the interests, influence, and impact of various stakeholders on the decision's success.",
    "Theory of Constraints": "Evaluate [my business decision] using the Theory of Constraints. Identify the limiting factor that impacts the overall process and optimize it to achieve better outcomes.",
    "Inverted Thinking": "Use Inverted Thinking to analyze [my business decision]. Instead of seeking solutions, identify ways to cause the problem and then work on preventing those causes.",
    "Sunk Cost Fallacy": "Apply awareness of the Sunk Cost Fallacy to assess [my business decision]. Avoid considering past investments and focus solely on the decision's future potential.",
    "Power-Interest Grid": "Analyze [my business decision] through the Power-Interest Grid. Categorize stakeholders based on their level of power and interest to prioritize communication and engagement.",
    "Framing Effect": "Evaluate [my business decision] considering the Framing Effect. Recognize how the presentation of information can influence decision-making and consider alternative framings.",
    "Incentive Alignment": "Utilize Incentive Alignment to assess [my business decision]. Ensure that the decision aligns the incentives of stakeholders towards the desired outcomes.",
    "Decision Fatigue": "Analyze [my business decision] while considering Decision Fatigue. Recognize how a series of decisions can lead to diminishing quality of subsequent choices."
}


# Iterate through each analysis model, send the prompt to GPT-4, and print the response
for analysis_type, prompt_template in prompts.items():
    prompt = prompt_template.replace("[my business decision]", decision)

    # Construct the messages for the chat completion
    messages = [{"role": "system", "content": "You are a business AI You are an engineering wizard, experienced at solving complex problems across various disciplines. Your knowledge is both wide and deep. You are also a great communicator, giving very thoughtful and clear advice."},
                {"role": "user", "content": prompt}]

    # Use the OpenAI API to generate a response from GPT-4 (replace "gpt-4" with the actual engine name when available)
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Update with the GPT-4 engine name when available
        messages=messages,
    )

    # Extract and print the response text
    result = response.choices[0].message['content']
    print(f"\n{analysis_type} Analysis:\n{result}\n")
