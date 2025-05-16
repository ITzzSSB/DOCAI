import os
from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini with correct model name and proper safety settings
llm = ChatGoogleGenerativeAI(
    model="gemini-1.0-pro",
    temperature=0.7,
    safety_settings={
        "HARM_CATEGORY_HARASSMENT": 1,  # BLOCK_NONE
        "HARM_CATEGORY_HATE_SPEECH": 1,  # BLOCK_NONE
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": 1,  # BLOCK_NONE
        "HARM_CATEGORY_DANGEROUS_CONTENT": 1,  # BLOCK_NONE
    }
)

# Initialize Tavily
tavily_tool = TavilySearchResults(max_results=3)

@tool
def check_symptoms(symptoms: str) -> str:
    """Check if the symptoms require immediate medical attention or can be treated at home."""
    prompt = """As a medical professional, analyze these symptoms:
{symptoms}

Respond with ONLY one of these exact phrases:
- 'URGENT' if emergency care is needed
- 'SEE_DOCTOR' if professional evaluation is recommended
- 'HOME_TREATMENT' if manageable with self-care""".format(symptoms=symptoms)
    
    try:
        response = llm.invoke(prompt).content
        return response.strip().upper()
    except Exception as e:
        print(f"Error in check_symptoms: {str(e)}")
        return "SEE_DOCTOR"  # Default to safest option

@tool
def recommend_medicine(symptoms: str) -> str:
    """Recommend medicines and dosages for non-urgent symptoms after verification."""
    urgency = check_symptoms.invoke({"symptoms": symptoms})
    
    if urgency == "URGENT":
        return "🆘 EMERGENCY: Please seek immediate medical attention or call emergency services."
    elif urgency == "SEE_DOCTOR":
        return "👨‍⚕️ Please consult a healthcare professional for proper evaluation."
    
    try:
        search_query = f"evidence-based OTC treatment guidelines for {symptoms} site:.gov OR site:.edu"
        search_results = tavily_tool.invoke({"query": search_query})
        
        prompt = """As a medical advisor, provide recommendations for:
Symptoms: {symptoms}

Guidelines: {search_results}

Include:
1. Recommended OTC medications (generic names)
2. Standard adult dosages
3. Contraindications
4. When to seek medical help

Format clearly with bullet points.""".format(symptoms=symptoms, search_results=search_results)
        
        response = llm.invoke(prompt).content
        return f"💊 Treatment Suggestions:\n{response}"
    except Exception as e:
        print(f"Error in recommend_medicine: {str(e)}")
        return "I couldn't retrieve treatment information. Please consult a pharmacist or doctor."

class AgentState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage, SystemMessage]], "messages"]

def create_doctor_agent():
    tools = [check_symptoms, recommend_medicine, tavily_tool]
    
    system_prompt = '''You are DoctorAI, a virtual medical assistant. Follow these rules:
1. First assess symptom urgency using check_symptoms
2. For emergencies, direct to immediate care
3. For non-emergencies, provide:
   - General advice
   - OTC options (when asked)
   - Warning signs
4. Never diagnose or prescribe
5. Always recommend professional care for serious concerns
6. Maintain empathetic, professional tone

Start by asking: "What symptoms are you experiencing?"'''
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    return agent_executor

def process_message(agent_executor, message: str, chat_history: list) -> str:
    try:
        messages = []
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        messages.append(HumanMessage(content=message))
        
        response = agent_executor.invoke({"messages": messages})
        return response["output"]
    except Exception as e:
        print(f"Error in process_message: {str(e)}")
        return "I encountered an error processing your request. Please try again or consult a healthcare professional."

# Initialize agent with error handling
try:
    doctor_agent = create_doctor_agent()
except Exception as e:
    print(f"Failed to initialize agent: {str(e)}")
    raise