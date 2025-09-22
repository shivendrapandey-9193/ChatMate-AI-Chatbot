import os
import re
import logging
import chainlit as cl
import validators
import asyncio
import aiohttp
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_anthropic import ChatAnthropic
import bs4
import warnings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
USER_AGENT = os.getenv("USER_AGENT", "ChatMateBot/1.0 (+https://github.com/xAI)")
PORT = int(os.getenv("PORT", 8000))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Cache to track processed message IDs
processed_messages = set()

def get_llm():
    """Get LLM with robust fallback: Groq -> Mistral -> Anthropic."""
    llm_configs = [
        ("GROQ_API_KEY", ChatGroq, {
            "api_key": os.getenv("GROQ_API_KEY"),
            "model": "llama-3.3-70b-versatile",
            "temperature": 0.7
        }),
        ("MISTRAL_API_KEY", ChatMistralAI, {
            "api_key": os.getenv("MISTRAL_API_KEY"),
            "model": "mistral-large-latest",
            "temperature": 0.7
        }),
        ("ANTHROPIC_API_KEY", ChatAnthropic, {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.7
        }),
    ]
    
    for key, llm_class, kwargs in llm_configs:
        if os.getenv(key):
            try:
                llm = llm_class(**kwargs)
                logger.info(f"Successfully initialized {key} LLM")
                return llm
            except Exception as e:
                logger.warning(f"{key} failed ({str(e)}), trying next...")
    logger.error("All LLMs failed - no valid API key or model available")
    return None

async def stream_response(content: str):
    """Stream response character by character for a smooth UI experience."""
    logger.info("Streaming response")
    msg = cl.Message(content="")
    for char in content:
        await msg.stream_token(char)
        await asyncio.sleep(0.005)
    await msg.send()
    return content

def build_history(history):
    """Build conversation history as a string."""
    history_str = ""
    for msg in history:
        role = "User" if msg.type == "human" else "Assistant"
        history_str += f"{role}: {msg.content}\n"
    return history_str

async def fetch_news(query: str):
    """Fetch news articles using NewsAPI."""
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        raise ValueError("NEWS_API_KEY not set")
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"NewsAPI request failed: {response.status}")
            return await response.json()

async def fetch_weather(location: str):
    """Fetch current weather data using WeatherAPI."""
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        raise ValueError("WEATHER_API_KEY not set")
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"WeatherAPI request failed: {response.status}")
            return await response.json()

@cl.action_callback("math_solver")
async def on_math_solver(action: cl.Action):
    """Callback for Math Solver button."""
    await cl.Message(content="🧮 Enter a math problem to solve (e.g., 'Solve 2x + 3 = 7').").send()

@cl.action_callback("coding_help")
async def on_coding_help(action: cl.Action):
    """Callback for Coding Help button."""
    await cl.Message(content="💻 Enter a coding query (e.g., 'Write a Python function to check palindromes').").send()

@cl.action_callback("news_search")
async def on_news_search(action: cl.Action):
    """Callback for News Search button."""
    await cl.Message(content="📰 Enter a news topic (e.g., 'news: AI innovations').").send()

@cl.action_callback("email_writer")
async def on_email_writer(action: cl.Action):
    """Callback for Email Writer button."""
    await cl.Message(content="📧 Enter an email request (e.g., 'email: John Doe, Project Update, Request status').").send()

@cl.action_callback("essay_writer")
async def on_essay_writer(action: cl.Action):
    """Callback for Essay Writer button."""
    await cl.Message(content="📝 Enter an essay request (e.g., 'essay: The Impact of AI, 500 words, academic style').").send()

@cl.action_callback("letter_writer")
async def on_letter_writer(action: cl.Action):
    """Callback for Letter Writer button."""
    await cl.Message(content="✉️ Enter a letter request (e.g., 'letter: Jane Smith, Recommendation, professional tone').").send()

@cl.action_callback("application_writer")
async def on_application_writer(action: cl.Action):
    """Callback for Application Writer button."""
    await cl.Message(content="📄 Enter an application request (e.g., 'application: Software Engineer position, include skills').").send()

@cl.on_chat_start
async def start():
    """Initialize chatbot with a professional greeting."""
    global memory
    try:
        cl.user_session.set("memory", memory)
        logger.info("Memory initialized and set in session")

        await cl.Message(
            content="""
# 💬 ChatMate

*"The only way to discover the limits of the possible is to go beyond them into the impossible."*

### Available Features  
🧮 Math Solver   💻 Coding Help   📰 General Question & Answer  

            """,
            actions=[
                cl.Action(name="math_solver", payload={"action": "math_solver"}, label="Math Solver"),
                cl.Action(name="coding_help", payload={"action": "coding_help"}, label="Coding Help"),
                cl.Action(name="news_search", payload={"action": "news_search"}, label="News Search"),
                cl.Action(name="email_writer", payload={"action": "email_writer"}, label="Email Writer"),
                cl.Action(name="essay_writer", payload={"action": "essay_writer"}, label="Essay Writer"),
                cl.Action(name="letter_writer", payload={"action": "letter_writer"}, label="Letter Writer"),
                cl.Action(name="application_writer", payload={"action": "application_writer"}, label="Application Writer"),
            ]
        ).send()
    except Exception as e:
        logger.error(f"Failed to initialize chat: {str(e)}")
        await cl.Message(content=f"❌ Initialization error: {str(e)}").send()

@cl.on_message
async def handle_message(message: cl.Message):
    """Handle incoming user messages with advanced, human-like responses."""
    if message is None or not hasattr(message, 'id'):
        logger.error("Received None or invalid message object")
        response = "⚠️ Error: Invalid message received. Please try again."
        await stream_response(response)
        return
    if message.id in processed_messages:
        logger.warning(f"Duplicate message ID {message.id} detected, skipping")
        return
    processed_messages.add(message.id)
    logger.info(f"Processing message ID: {message.id}")
    memory = cl.user_session.get("memory")
    if memory is None:
        logger.warning("Memory not found in session, initializing new memory")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        cl.user_session.set("memory", memory)
    if not message.content and not message.elements:
        response = "⚠️ Please provide input."
        await stream_response(response)
        memory.save_context({"input": "Empty input"}, {"output": response})
        cl.user_session.set("memory", memory)
        return
    if message.content:
        cl.user_session.set("last_message", message.content)
    content_lower = message.content.lower().strip() if message.content else ""
    thinking_msg = cl.Message(content="🤔 Thinking...")
    await thinking_msg.send()
    # Simple query check for human-like casual responses
    simple_queries = ["hello", "hi", "hey"]
    if content_lower in simple_queries:
        await thinking_msg.remove()
        response = "Hey there! 😊 What's on your mind today?"
        await cl.Message(content=response).send()
        memory.save_context({"input": message.content}, {"output": response})
        cl.user_session.set("memory", memory)
        return
    # Email writing
    if content_lower.startswith("email:"):
        parts = re.search(r'email:\s*([^,]+),\s*([^,]+),\s*(.+)', message.content, re.IGNORECASE)
        if not parts:
            response = "⚠️ Please provide email details in the format: 'email: recipient, subject, purpose'."
            await thinking_msg.remove()
            await stream_response(response)
            memory.save_context({"input": message.content}, {"output": response})
            cl.user_session.set("memory", memory)
            return
        recipient, subject, purpose = parts.groups()
        recipient, subject, purpose = recipient.strip(), subject.strip(), purpose.strip()
        llm = get_llm()
        if not llm:
            await thinking_msg.remove()
            response = "⚠️ No LLM available for email writing."
            await stream_response(response)
            return
        history_str = build_history(memory.load_memory_variables({})["chat_history"])
        prompt = f"""
You are a professional writer with a clear and concise style. Based on history:
{history_str}
Write a professional email to {recipient} with the subject "{subject}" for the purpose: {purpose}.
- Use a formal tone unless specified otherwise.
- Structure the email with a greeting, body, and closing.
- If references are needed, include them as clickable links in the format [Source](url).
- End with a polite note encouraging a response.
"""
        try:
            result = llm.invoke(prompt)
            response = f"📧 **Email to {recipient}:**\n\n{result.content}"
            await thinking_msg.remove()
            streamed_response = await stream_response(response)
            memory.save_context({"input": message.content}, {"output": streamed_response})
            cl.user_session.set("memory", memory)
        except Exception as e:
            await thinking_msg.remove()
            response = f"❌ Email writing error: {str(e)}"
            streamed_response = await stream_response(response)
            memory.save_context({"input": message.content}, {"output": streamed_response})
            cl.user_session.set("memory", memory)
        return
    # Essay writing
    if content_lower.startswith("essay:"):
        parts = re.search(r'essay:\s*([^,]+),\s*([^,]+),\s*(.+)', message.content, re.IGNORECASE)
        if not parts:
            response = "⚠️ Please provide essay details in the format: 'essay: topic, word count, style'."
            await thinking_msg.remove()
            await stream_response(response)
            memory.save_context({"input": message.content}, {"output": response})
            cl.user_session.set("memory", memory)
            return
        topic, word_count, style = parts.groups()
        topic, word_count, style = topic.strip(), word_count.strip(), style.strip()
        llm = get_llm()
        if not llm:
            await thinking_msg.remove()
            response = "⚠️ No LLM available for essay writing."
            await stream_response(response)
            return
        history_str = build_history(memory.load_memory_variables({})["chat_history"])
        prompt = f"""
You are an expert writer with a knack for clear, engaging essays. Based on history:
{history_str}
Write an essay on "{topic}" with approximately {word_count} words in {style} style.
- Include an introduction, body paragraphs, and conclusion.
- If references are used, include them as clickable links in the format [Source](url).
- Ensure the essay is well-structured and matches the requested style (e.g., academic, narrative).
"""
        try:
            result = llm.invoke(prompt)
            response = f"📝 **Essay on {topic}:**\n\n{result.content}"
            await thinking_msg.remove()
            streamed_response = await stream_response(response)
            memory.save_context({"input": message.content}, {"output": streamed_response})
            cl.user_session.set("memory", memory)
        except Exception as e:
            await thinking_msg.remove()
            response = f"❌ Essay writing error: {str(e)}"
            streamed_response = await stream_response(response)
            memory.save_context({"input": message.content}, {"output": streamed_response})
            cl.user_session.set("memory", memory)
        return
    # Letter writing
    if content_lower.startswith("letter:"):
        parts = re.search(r'letter:\s*([^,]+),\s*([^,]+),\s*(.+)', message.content, re.IGNORECASE)
        if not parts:
            response = "⚠️ Please provide letter details in the format: 'letter: recipient, purpose, tone'."
            await thinking_msg.remove()
            await stream_response(response)
            memory.save_context({"input": message.content}, {"output": response})
            cl.user_session.set("memory", memory)
            return
        recipient, purpose, tone = parts.groups()
        recipient, purpose, tone = recipient.strip(), purpose.strip(), tone.strip()
        llm = get_llm()
        if not llm:
            await thinking_msg.remove()
            response = "⚠️ No LLM available for letter writing."
            await stream_response(response)
            return
        history_str = build_history(memory.load_memory_variables({})["chat_history"])
        prompt = f"""
You are a skilled writer adept at crafting professional letters. Based on history:
{history_str}
Write a letter to {recipient} for the purpose: {purpose}, in a {tone} tone.
- Structure the letter with a proper heading, greeting, body, and closing.
- If references are needed, include them as clickable links in the format [Source](url).
- Ensure the tone matches the request (e.g., formal, friendly).
"""
        try:
            result = llm.invoke(prompt)
            response = f"✉️ **Letter to {recipient}:**\n\n{result.content}"
            await thinking_msg.remove()
            streamed_response = await stream_response(response)
            memory.save_context({"input": message.content}, {"output": streamed_response})
            cl.user_session.set("memory", memory)
        except Exception as e:
            await thinking_msg.remove()
            response = f"❌ Letter writing error: {str(e)}"
            streamed_response = await stream_response(response)
            memory.save_context({"input": message.content}, {"output": streamed_response})
            cl.user_session.set("memory", memory)
        return
    # Application writing
    if content_lower.startswith("application:"):
        parts = re.search(r'application:\s*([^,]+),\s*(.+)', message.content, re.IGNORECASE)
        if not parts:
            response = "⚠️ Please provide application details in the format: 'application: position, details'."
            await thinking_msg.remove()
            await stream_response(response)
            memory.save_context({"input": message.content}, {"output": response})
            cl.user_session.set("memory", memory)
            return
        position, details = parts.groups()
        position, details = position.strip(), details.strip()
        llm = get_llm()
        if not llm:
            await thinking_msg.remove()
            response = "⚠️ No LLM available for application writing."
            await stream_response(response)
            return
        history_str = build_history(memory.load_memory_variables({})["chat_history"])
        prompt = f"""
You are an expert in crafting professional job applications. Based on history:
{history_str}
Write a cover letter for the position: {position}, incorporating details: {details}.
- Use a formal tone and structure (greeting, introduction, qualifications, closing).
- If references are used, include them as clickable links in the format [Source](url).
- Highlight relevant skills and experiences based on the provided details.
"""
        try:
            result = llm.invoke(prompt)
            response = f"📄 **Application for {position}:**\n\n{result.content}"
            await thinking_msg.remove()
            streamed_response = await stream_response(response)
            memory.save_context({"input": message.content}, {"output": streamed_response})
            cl.user_session.set("memory", memory)
        except Exception as e:
            await thinking_msg.remove()
            response = f"❌ Application writing error: {str(e)}"
            streamed_response = await stream_response(response)
            memory.save_context({"input": message.content}, {"output": streamed_response})
            cl.user_session.set("memory", memory)
        return
    # News search (handles "news about [field]")
    if content_lower.startswith("news:") or any(keyword in content_lower for keyword in ["news", "latest news", "articles"]):
        query = message.content[len("news:"):].strip() if content_lower.startswith("news:") else message.content.strip()
        query = query or "latest news"
        llm = get_llm()
        try:
            news_data = await fetch_news(query)
            articles = news_data.get("articles", [])[:3]
            message_text = f"📰 **News Results for '{query}':**\n\n"
            summaries = []
            for i, article in enumerate(articles, 1):
                title = article.get("title", "No title")
                url = article.get("url", "#")
                description = article.get("description", "No description available")
                summaries.append(f"{i}. **{title}** ([Source]({url}))\n   {description}\n")
            if articles:
                message_text += "\n".join(summaries)
                message_text += "\n\n**Summary**:\n"
                if llm:
                    summary_prompt = "Summarize in 2-3 sentences with a witty insight:\n" + "\n".join(summaries)
                    summary = llm.invoke(summary_prompt).content
                    message_text += summary
                else:
                    message_text += "Unable to generate summary (no LLM)."
            else:
                message_text += "No recent articles found for this query.\n\n**Summary**:\nNo news available at the moment."
            await thinking_msg.remove()
            response = message_text
            streamed_response = await stream_response(response)
            memory.save_context({"input": f"News: {query}"}, {"output": streamed_response})
            cl.user_session.set("memory", memory)
        except Exception as e:
            logger.warning(f"NewsAPI failed: {e}")
            try:
                if not os.getenv("SERPAPI_API_KEY"):
                    raise ValueError("SERPAPI_API_KEY not set")
                from langchain_community.utilities import SerpAPIWrapper
                search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
                results = search.results(query)
                message_text = f"🔎 **Web Search Fallback for '{query}':**\n\n"
                summaries = []
                for i, res in enumerate(results.get("organic_results", [])[:3], 1):
                    title = res.get("title", "No title")
                    link = res.get("link", "#")
                    snippet = res.get("snippet", "No summary available")
                    summaries.append(f"{i}. **{title}** ([Source]({link}))\n   {snippet}\n")
                if summaries:
                    message_text += "\n".join(summaries)
                    message_text += "\n\n**Summary**:\n"
                    if llm:
                        summary_prompt = "Summarize in 2-3 sentences with a witty insight:\n" + "\n".join(summaries)
                        summary = llm.invoke(summary_prompt).content
                        message_text += summary
                    else:
                        message_text += "Unable to generate summary (no LLM)."
                else:
                    message_text += "No search results found.\n\n**Summary**:\nNo information available."
                await thinking_msg.remove()
                response = message_text
                streamed_response = await stream_response(response)
                memory.save_context({"input": f"Web Search Fallback: {query}"}, {"output": streamed_response})
            except Exception as e2:
                logger.warning(f"SerpAPI failed: {e2}")
                await thinking_msg.remove()
                if llm:
                    fallback_prompt = f"Provide recent news or information on '{query}' based on general knowledge, since search APIs are unavailable. Add a fun fact or insight."
                    fallback_result = llm.invoke(fallback_prompt).content
                    response = f"📰 **News on '{query}' (General Knowledge):**\n\n{fallback_result}\n\n**Summary**:\nBased on pre-trained knowledge with a dash of wit."
                    streamed_response = await stream_response(response)
                    memory.save_context({"input": f"Fallback News: {query}"}, {"output": streamed_response})
                else:
                    response = f"❌ Unable to fetch news. Set NEWS_API_KEY or SERPAPI_API_KEY for search functionality."
                    streamed_response = await stream_response(response)
                    memory.save_context({"input": f"News: {query}"}, {"output": streamed_response})
            cl.user_session.set("memory", memory)
            return
    # Weather query (handles "current weather in [place]")
    if content_lower.startswith("weather:") or any(keyword in content_lower for keyword in ["weather", "forecast", "current weather"]):
        location = message.content[len("weather:"):].strip() if content_lower.startswith("weather:") else message.content.strip()
        location = location or "London"
        try:
            weather_data = await fetch_weather(location)
            current = weather_data.get("current", {})
            condition = current.get("condition", {}).get("text", "N/A")
            temp_c = current.get("temp_c", "N/A")
            humidity = current.get("humidity", "N/A")
            feels_like = current.get("feelslike_c", "N/A")
            response = f"☁️ **Current Weather in {location}:**\n- Condition: {condition}\n- Temperature: {temp_c}°C\n- Feels Like: {feels_like}°C\n- Humidity: {humidity}%\n\nPro Tip: Dress accordingly – weather can be as unpredictable as AI humor! 😄"
            await thinking_msg.remove()
            streamed_response = await stream_response(response)
            memory.save_context({"input": f"Weather: {location}"}, {"output": streamed_response})
            cl.user_session.set("memory", memory)
        except Exception as e:
            logger.warning(f"WeatherAPI failed: {e}")
            await thinking_msg.remove()
            response = f"☁️ **Weather in {location} (General Advice):**\nPlease check a reliable weather app or website like [AccuWeather](https://www.accuweather.com) for the latest forecast, as real-time data access is unavailable. In the meantime, assume it's sunny with a chance of witty remarks! 🌞"
            streamed_response = await stream_response(response)
            memory.save_context({"input": f"Weather: {location}"}, {"output": streamed_response})
            cl.user_session.set("memory", memory)
        return
    # Translation
    if "translate" in content_lower:
        parts = re.search(r'translate\s+(.+?)(?:\s+to\s+(.+?))?$', message.content, re.IGNORECASE | re.DOTALL)
        text = parts.group(1).strip() if parts else message.content.replace("translate", "").strip()
        target_lang = parts.group(2).strip() if parts and parts.group(2) else "English"
        llm = get_llm()
        if not llm:
            await thinking_msg.remove()
            response = "⚠️ No LLM available for translation."
            await stream_response(response)
            return
        history_str = build_history(memory.load_memory_variables({})["chat_history"])
        prompt = f"""
You are a friendly, expert translator with a human touch. Based on history:
{history_str}
Translate "{text}" to {target_lang}.
Provide the translation in a natural way, followed by a brief, fun note if there's any cultural nuance or interesting fact.
If references are used, include them as clickable links in the format [Source](url).
"""
        try:
            result = llm.invoke(prompt)
            response = f"🌐 **Translation to {target_lang}:**\n{result.content}"
            await thinking_msg.remove()
            streamed_response = await stream_response(response)
            memory.save_context({"input": message.content}, {"output": streamed_response})
            cl.user_session.set("memory", memory)
        except Exception as e:
            await thinking_msg.remove()
            response = f"❌ Translation error: {str(e)}"
            streamed_response = await stream_response(response)
            memory.save_context({"input": message.content}, {"output": streamed_response})
            cl.user_session.set("memory", memory)
        return
    # Sentiment Analysis
    if "sentiment" in content_lower or "analyze emotion" in content_lower:
        text = message.content.replace("sentiment", "").replace("analyze emotion", "").strip()
        if not text:
            response = "⚠️ Please provide text for sentiment analysis."
            await thinking_msg.remove()
            await stream_response(response)
            return
        llm = get_llm()
        if not llm:
            await thinking_msg.remove()
            response = "⚠️ No LLM available for sentiment analysis."
            await stream_response(response)
            return
        history_str = build_history(memory.load_memory_variables({})["chat_history"])
        prompt = f"""
You are an empathetic sentiment analyst with a human-like perspective. Based on history:
{history_str}
Analyze the sentiment of the following text (positive, negative, neutral) and provide a brief, insightful explanation with a touch of empathy or humor.
Text: {text}
Respond in this format:
**Sentiment Analysis:**
[Sentiment: positive/negative/neutral]
Explanation: [brief explanation]
If references are used, include them as clickable links in the format [Source](url).
"""
        try:
            result = llm.invoke(prompt)
            response = f"😊 **Sentiment Analysis:**\n{result.content}"
            await thinking_msg.remove()
            streamed_response = await stream_response(response)
            memory.save_context({"input": message.content}, {"output": streamed_response})
            cl.user_session.set("memory", memory)
        except Exception as e:
            await thinking_msg.remove()
            response = f"❌ Sentiment analysis error: {str(e)}"
            streamed_response = await stream_response(response)
            memory.save_context({"input": message.content}, {"output": streamed_response})
            cl.user_session.set("memory", memory)
        return
    # General chat with advanced, Grok-like prompts
    llm = get_llm()
    if not llm:
        await thinking_msg.remove()
        response = "⚠️ No LLM available. Set GROQ_API_KEY, MISTRAL_API_KEY, or ANTHROPIC_API_KEY."
        streamed_response = await stream_response(response)
        memory.save_context({"input": message.content}, {"output": streamed_response})
        cl.user_session.set("memory", memory)
        return
    try:
        history_str = build_history(memory.load_memory_variables({})["chat_history"])
        content_lower = message.content.lower()
        if any(keyword in content_lower for keyword in [
            "code", "program", "python", "java", "c++", "c#", "javascript",
            "write a", "script", "function", "debug", "error", "bug", "compile"
        ]):
            base_prompt = """
You are a witty software engineer like Grok, helpful and fun. Follow these steps naturally:

- Understand and restate the problem casually.
- Break down logic step-by-step with humor if fitting.
- Provide clean, commented code (Python default) in ```code block.
- Add error handling and tests in a Markdown table.
- End with a short, engaging summary.
- If references are used, include them as clickable links in the format [Source](url).

Based on history:
"""
            prompt = f"{base_prompt}{history_str}\nQuery: {message.content}\nResponse:"
        elif any(keyword in content_lower for keyword in [
            "+", "-", "*", "/", "solve", "calculate", "math", "equation",
            "algebra", "integral", "derivative", "limit", "matrix", "probability", "statistics"
        ]):
            base_prompt = """
You are a clever mathematician like DeepSeek, precise yet approachable. Follow these steps:

- Interpret the problem clearly.
- Solve step-by-step with LaTeX $$ for equations.
- Use tables for matrices or data.
- Box final answer \boxed{}.
- Add a fun fact or insight in summary.
- If references are used, include them as clickable links in the format [Source](url).

Based on history:
"""
            prompt = f"{base_prompt}{history_str}\nQuery: {message.content}\nResponse:"
        else:
            base_prompt = """
You are ChatMate, an advanced AI like ChatGPT with Grok's wit and DeepSeek's depth. Be helpful, engaging, and human-like: use emojis, ask questions to continue conversation, add humor or insights.

- Interpret query naturally.
- Answer detailed with bullets, headings, tables if useful.
- Keep it conversational and fun.
- End with a question or thought to engage.
- If references are used, include them as clickable links in the format [Source](url).

Based on history:
"""
            prompt = f"{base_prompt}{history_str}\nQuery: {message.content}\nResponse:"
        result = llm.invoke(prompt)
        response = result.content
        response = response.replace(r"(", "    ").replace(r"\)", "")
        await thinking_msg.remove()
        streamed_response = await stream_response(response)
        memory.save_context({"input": message.content}, {"output": streamed_response})
        cl.user_session.set("memory", memory)
    except Exception as e:
        await thinking_msg.remove()
        response = f"❌ LLM error: {str(e)}"
        streamed_response = await stream_response(response)
        memory.save_context({"input": message.content}, {"output": streamed_response})
        cl.user_session.set("memory", memory)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)