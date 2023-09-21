import os
import openai
import argparse
import random
from tqdm import tqdm
import time

from constants.metadata import DEFAULT_PROMPT_DIR
from constants.prompt_generation import DEFAULT_TEMPLATE, DEFAULT_TEMPERATURE, DEFAULT_AVAILABLE_M, DEFAULT_SWITCHING_PART

from dotenv import load_dotenv
load_dotenv()

def set_api_key():
    # Export API Key Environment Variables and save under OpenAI Instance
    openai.organization = os.getenv("OPENAI_API_ORG_KEY")
    openai.api_key = os.getenv("OPENAI_API_KEY")



def generate_prompts(
        categories, 
        save_dir, 
        num_prompts,
        verbose
    ):
    # print log
    if verbose:
        print(f"\nRunning for {len(categories)} number of categories...\n"
              f"Saving prompts under \"{save_dir}\"...\n\n")
    
    # Iterate through all categories
    for category in categories:
        # create saving directory
        os.makedirs(save_dir, exist_ok=True)

        # if m is None, randomly choose between 3~20
        if num_prompts is None:
            m = random.choice(DEFAULT_AVAILABLE_M)
        else:
            m = num_prompts
            
        # Assert m to be positive integer
        assert (m >= 1 and type(m) == int) or m is None, "Only positive integer allowed."
        
        # choose which to switch
        switch = random.choice(DEFAULT_SWITCHING_PART)
        # Instantiated Template
        instantiated_template = DEFAULT_TEMPLATE.replace("[M]", f"{m}").replace("[CATEGORY]", category).replace("[SWITCHING-PART]", switch)
        if verbose:
            print(f"INSTANTIATED-DEFAULT_TEMPLATE: {instantiated_template}\n\n")


        # HOI prompts to fill in
        HOI_prompts = []

        pbar = tqdm(total=m, desc=f"Generating Prompts for: [Category '{category}']")
        while len(HOI_prompts) != m:
            # reset if iterating again
            HOI_prompts = []


            # Chat-Completion
            # NOTE: OpenAI API does not support random-seeding. 
            # Reference: https://community.openai.com/t/is-there-a-way-to-set-a-a-random-seed-for-responses-with-temperature-0/4164 
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user",
                    "content": instantiated_template}],
                temperature=DEFAULT_TEMPERATURE,
            )

            # iterate through all available lines
            for idx, HOI_prompt in enumerate(response['choices'][0].message['content'].split("\n")):
                # Remove any remaining whitespace or \n at the start-end of the sentence
                HOI_prompt = HOI_prompt.strip()

                # If there is number, remove the number
                if f"{idx+1}." in HOI_prompt[:3]:
                    # Remove the number and remaining whitespaces
                    HOI_prompt = HOI_prompt.replace(f"{idx+1}.", "").strip()
                    
                # If HOI prompt is a blank, continue
                if HOI_prompt == "":
                    continue
                
                # If the sentence ends with ".", remove it
                if HOI_prompt[-1] == ".":
                    # Remove the "." and remaining whitespaces
                    HOI_prompt = HOI_prompt[:-1].strip()

                # Add to HOI prompts
                HOI_prompts.append(HOI_prompt)
                pbar.update(1)

            if len(HOI_prompts) != m:
                pbar = tqdm(total=m, desc=f"Restart Category: {category}")
                continue
            else:
                if verbose:
                    print(f"=========CATEGORY: {category}=========")
                    print(response['choices'][0].message['content'])
                break

        # Save to txt file
        with open(os.path.join(save_dir, f"{category}.txt"), "w") as wf:
            wf.write("\n".join(HOI_prompts))
        
    



if __name__=="__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", type=str)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--num_prompts", type=int, default=None) # If None, m is randomly chosen between 3~20
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Set OpenAI API Key
    set_api_key()

    # generate HOI prompts
    generate_prompts(
        categories=args.categories, 
        save_dir=args.save_dir, 
        num_prompts=args.num_prompts,
        verbose=args.verbose
    )