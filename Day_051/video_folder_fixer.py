import os
import time
import openai
import re

# Define your API key
api_key = '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'  # Replace with your actual API key

# Set the directory path
directory_path = r"\\qnapmrbig\Media\Movies\__NEW"

# Initialize the OpenAI API
openai.api_key = api_key

# Define a function to sanitize the folder name to be Windows-friendly
def sanitize_folder_name(folder_name):
    # Remove any characters that are not allowed in Windows folder names
    # Windows does not allow <>:"/\|?* and non-printable characters in file names
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', folder_name)

# Define a function to get the cleaned name from OpenAI
def get_cleaned_name(folder_name):
    system_message = "You are a file organizing assistant. Provide a cleaned, organized name for a folder. Using underscores instead of spaces, and no special characters. as and example Pig_2021_1080p_WEB_DL_DD5_1_x264_EVO_TGx should be Pig_2021,  have the date at the end of the name if a year is provided, capatalize the names, remove anything like dvdrip, xvid, bluray, REPACK, DVDSCR, h264, XViD"
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Provide a clean name for this folder: '{folder_name}'"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the appropriate model
        messages=messages,
    )

    result = response.choices[0].message['content']
    cleaned_name = result.split(':', 1)[-1].strip().replace(' ', '_') if ':' in result else result.strip().replace(' ', '_')
    return cleaned_name

# List all subfolders in the directory
subfolders = [f.name for f in os.scandir(directory_path) if f.is_dir()]

# Dictionary to hold original and suggested names
renaming_plan = {}

# Iterate through subfolders and get suggestions
for folder_name in subfolders:
    print (folder_name)
    cleaned_name = get_cleaned_name(folder_name)
    sanitized_name = sanitize_folder_name(cleaned_name)
    print (sanitized_name)
    time.sleep(2)
    renaming_plan[folder_name] = sanitized_name

# Output the suggested renaming plan
for original, cleaned in renaming_plan.items():
    print(f"Original: {original}")
    print(f"Suggested: {cleaned}")
    print()

# Ask the user if they want to proceed with renaming
confirmation = input("Do you want to proceed with renaming the folders as suggested? (yes/no): ")
if confirmation.lower() == 'yes':
    for original, cleaned in renaming_plan.items():
        original_path = os.path.join(directory_path, original)
        cleaned_path = os.path.join(directory_path, cleaned)
        os.rename(original_path, cleaned_path)  # Rename the folder
        print(f"Renamed '{original}' to '{cleaned}'")
