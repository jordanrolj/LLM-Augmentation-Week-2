"""
A simple example of function calling with OpenAI.

"""

import openai
from dotenv import load_dotenv
import os
import requests

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_joke() -> str:
    # URL for a joke
    joke_api_url = "https://icanhazdadjoke.com/"

    # Set the headers to specify the response format as plain text
    headers = {
        'Accept': 'text/plain'
    }

    # Send the GET request
    response = requests.get(joke_api_url, headers=headers)
    
    # Check if the request was successful
    if response.status_code == 200:
        joke = response.text.strip()  # Return the joke text
        return joke
    else:
        raise Exception("Failed to retrieve joke")


    # Return a string the joke
    # Uncomment the following line to see the returned object
    print(str(response))

# Function to call the OpenAI API and get a response, whether it's a completion or a tool call
def get_completion(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=300, tools=None):
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools
    )
    return response.choices[0].message

if __name__ == "__main__":
    # Define the tools that we want to use
    # This is defined in JSON format for the OpenAI API
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_joke",
                "description": "Returns a random dad joke from icanhazdadjoke.com",
                # I should have included a better example... The parameters would go below
                "parameters": {
                    "type": "object",
                    "properties": {},  # No parameters needed for this request
                },
                #if there were required parameters for the api call, they would need to be included below
                "required": [],
            },
        }
    ]

    # Pre-load a message to begin the conversation
    msg = "Tell me a funny joke! Actually, Make that two!"
    messages = [
        {
            "role": "user",
            "content": msg
        }
    ]
    print(f"User: {msg}\n---")

    response = get_completion(messages, tools=tools)

    # Now, we need to parse the response - the response will contain a TOOL CALL, rather than a completion.
    # The TOOL CALL tells us the function (and appropriate arguments) that the LLM wants to call.
    # This works because OpenAI's API LLMs have been fine-tuned to understand and call functions - other LLMs, such as Llama, do not have this capability.

    # Uncomment the following line to see the response object
    #print(response)

    response = get_completion(messages, tools=tools)

    # Parse the tool call and execute the function
    tool_responses = []
    function_map = {
        "get_joke": get_joke
    }
    # multiple functions can be called at the same time, so we must account for that.
    for tool_call in response.tool_calls:
        function_call = tool_call.function.name
        function_args = eval(tool_call.function.arguments)
        print(f"Calling function: {function_call}")

        # Call the function and get the result
        tool_response = function_map[function_call]()
        print(f"Function returns: {tool_response}\n---")
        
        tool_responses.append({"function_name": function_call, "tool_response": tool_response})

    # Adjust the messages array for the next API call
    messages.append(response)

    for idx, tool_call in enumerate(response.tool_calls):
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_responses[idx]['function_name'],
                "content": tool_responses[idx]['tool_response'],
            }
        )

    # Call the completion API again, this time with the function response
    response = get_completion(messages, tools=tools)

    print(f"AI: {response.content}\n---")

