import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=8.0, max_retries=0)


def generate_narration(snapshot):
    fundraising_line = ""
    fundraising_result = snapshot.get("fundraising_result")
    if fundraising_result:
        if fundraising_result["raised"]:
            fundraising_line = f"\nThe founder attempted to raise funding this month and succeeded, raising ${fundraising_result['amount_raised']}."
        else:
            fundraising_line = "\nThe founder attempted to raise funding this month but was unsuccessful."

    prompt = f"""Write a short, 2-3 sentence summary of month {snapshot['month_number']} for a startup founder.
Only describe the facts given below. Do not invent or estimate any numbers not listed here.

Revenue: ${snapshot['revenue']}
Total customers: {snapshot['customer_count']}
New customers acquired this month: {snapshot['customers_acquired']}
Customers churned this month: {snapshot['customers_churned']}
Cash on hand: ${snapshot['cash_on_hand']}
Marketing spend: ${snapshot['marketing_spend']}
Employees: {snapshot['employee_count']}
Overall market conditions this month: {snapshot['market_condition']}{fundraising_line}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
