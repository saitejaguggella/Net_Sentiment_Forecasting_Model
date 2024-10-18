import streamlit as st
import pandas as pd
import re
import os
import pandas as pd
from openai import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Load the CSV file
df = pd.read_csv("Cleaned_Copilot+PC's_with_specs.csv")

# Set environment variables for OpenAI API
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

deployment_name = 'Surface_Analytics'

start_phrase = """

You are an AI Assistant, your task is to extract relevant sub-aspects from below mentioned list of Aspect and Sub-Aspects and their corresponding sentiment (Positive or Negative) from the user's prompt. This information will be used to forecast user sentiment accurately.

Please carefully analyze the user's prompt to identify the appropriate sub-aspects from the predefined list provided below.

### Instructions:
1. Identify only Relevant Sub-Aspects: Based on the user's prompt, select the most appropriate sub-aspects from the predefined list.
2. Assign Sentiments: Classify each sub-aspect as either Positive or Negative. Make sure:
   - Sub-aspects are unique within each sentiment category.
3. IMPORTANT: Ensure that no sub-aspect is repeated in both the Positive and Negative categories under any circumstances.
4. Handle Irrelevant Prompts: If the prompt is unrelated to any predefined sub-aspects, mark both Positive and Negative categories as "None."

*Note: Thoroughly examine each prompt for accurate extraction like for example if in user prompt he mentioned about any aspect which will affect the net sentiment less then you should mention those related keywords in Negative, as these sub-aspects are essential for reliable sentiment forecasting.*

### Example Format:
- User Prompt: "I have increased the performance for the Surface Laptop Pro 11 and decreased the display size, so can you generate hypothetical reviews?"
  
- Expected Answer Format:
  - Positive: Processor/CPU Related, Thermal/Temperature,Efficiency, Optimization, Improvement, Upgrades, Comparison, Benchmarking, Productivity, Work, Testing, Development, Thermal Management, Hardware, Connectivity,Ports, Storage/Memory.
  - Negative: Display, Size, Price, Size and Form Factor, Brightness and Visibility.


### Predefined List of Aspects and Sub-Aspects:
- Performance: Performance, Processor/CPU Related, Thermal/Temperature, Power, Energy, Gaming, Graphics, Efficiency, Optimization, Issues, Failures, Improvement, Upgrades, Comparison, Benchmarking, Productivity, Work, Testing, Development, Thermal Management, Memory, Storage.
- Design: Aesthetics, Appearance, Size, Weight, Build Quality, Durability, Materials, Finishes, Portability, Flexibility, Design, Form Factor, Customization, Personalization, Branding, Logos, Interaction, Innovation, Technology, Usability, Comfort, Environmental, Sustainability, Accessories, Add-ons, Ventilation, Cooling.
- Software: Operating Systems, Development, Programming, Software Applications, System Management Tools, Software Functionality, Features, Multimedia, Creative Tools, Web & Cloud Services, Database, Data Management, Security, Privacy, Automation, AI, Productivity Tools, Performance, Optimization, Design, User Experience, Learning, Development, Integration, Interoperability, Hardware, Devices, Miscellaneous.
- AI Capabilities: Core AI Technologies and Concepts, AI Applications and Capabilities, AI Hardware and Processing, AI Platforms and Tools, AI Functions and Performance, AI in Development and Engineering, AI Use Cases and Applications, AI in Business and Marketing, AI Ethics and Risks, AI Hype and Buzzwords, Miscellaneous Technologies, Common AI Keywords.
- Display: Display Technology Types, Resolution and Image Quality, Size and Form Factor, Color and Visual Quality, Touch and Interaction, Brightness and Visibility, Refresh Rate and Performance, Display Features and Enhancements, Screen Design and Build, Display Modes and Settings, Usage and Viewing Experience, Issues and Drawbacks, Advanced Features and Specs, General and Descriptive Terms.
- Co-Creator: Collaboration & Communication, Creativity & Innovation, Tools & Technology, Support & Guidance, Productivity & Efficiency, Content Creation & Sharing, Business & Partnerships, Innovation & Future, Design & Development.
- Battery: Battery Life & Longevity, Power Consumption & Efficiency, Charging & Power Supply, Performance & Usage, Size & Capacity, Additional Features & Characteristics.
- Hardware: Processor & CPU, Memory & Storage, Hardware Components & Build, Brands & Models, Performance & Specifications, Components & Accessories, Form Factors & Design, Manufacturing & Technology, Miscellaneous.
- Security: Authentication and Access Control, Data Protection and Privacy, Cybersecurity and Threats, System Security and Integrity, Management and Monitoring, Encryption and Data Storage, Security Practices and Tools, Security Vulnerabilities and Risks, Regulatory and Compliance, Encryption and Authentication Technologies.
- Price: Pricing and Cost Overview, Discounts, Deals, and Savings, Expense and Investment, Price Changes and Adjustments, Comparison and Valuation, Budgeting and Affordability, Sales and Revenue, Special Terms and Conditions, Financial Management and Tools.
- Connectivity: Connection Types and Technologies, Connection Methods and Devices, Connectivity and Integration, Data Transfer and Communication, Network and Signal, Compatibility and Access, Connectivity Issues and Support, Smart and Advanced Features, Miscellaneous and Additional Terms.
- Gaming: Gaming Platforms and Devices, Performance and Hardware, Gaming Genres and Types, Gaming Experiences and Features, Game Titles and Franchises, Game Development and Customization, Gaming Accessories and Equipment, Gaming Terminology and Metrics, Community and Social Aspects, Game Modes and Play Styles, Miscellaneous.
- Audio: Audio Devices and Equipment, Audio Quality and Characteristics, Audio Effects and Features, Audio Playback and Usage, Audio Technologies and Brands, Audio Characteristics and Performance, Audio Settings and Adjustments, Audio Descriptors and Qualifiers, Audio Interactions and Community, Miscellaneous.
- Keyboard: Keyboard Types and Layouts, Keyboard Features and Functionality, Comfort and Usability, Design and Aesthetics, Performance and Durability, Size and Form Factor, Adjustment and Customization, Additional Features and Accessories, Feedback and User Experience, Miscellaneous.
- Ports: USB Types and Variants, Connections and Compatibility, Ports and Slots, Design and Placement, Performance and Usability, Types of Ports, Miscellaneous.
- Graphics: Graphics Hardware and Models, Graphics Performance and Quality, Graphics Processing and Technology, Graphics Cards and Features, Visual Effects and Quality, Testing and Benchmarking, Drivers and Updates, Configuration and Settings, Miscellaneous.
- Recall: Recall and Memory Processes, Search and Retrieval Techniques, Information Handling and Management, Accuracy and Verification, Communication and Documentation, Issue and Problem Handling, Search Tools and Technology, Data Analysis and Processing, Recovery and Restoration, Memory and Cognitive Processes, Miscellaneous Concepts, Security and Privacy.
- Touchpad: Touchpad Functionality and Performance, Touchpad Size and Shape, Touchpad Features and Gestures, Touchpad Materials and Design, Touchpad Issues and Problems, Touchpad Interaction and Feedback, Touchpad Usability and Customization, Touchpad Aesthetics and Design Considerations, Touchpad Enhancements and Adjustments, Touchpad Interaction with Other Devices, Touchpad Quality and Feedback, Touchpad Adjustments and Controls.
- Automatic Super Resolution: Resolution and Quality, Generation and Enhancement, Features and Tools, Visual Effects and Processing, Detection and Analysis, Technical Aspects, Miscellaneous, Quality Improvement and Features.
- Camera: Camera Types and Features, Image Quality and Resolution, Camera Functionality and Controls, Technical and Image Processing, Visual Effects and Improvements, Camera Design and Build, Miscellaneous.
- Live Captions: Live Captions and Subtitles, Translation and Transcription, Video Streaming, Functionality and Features, Communication and Interaction, Accessibility and Usability, Editing and Quality, Technical and System Aspects, Issues and Problems, Additional Context and Information.
- Account: Account Access and Authentication, Account MaCnagement and Configuration, Communication and Support, Privacy and Security, Billing and Financials, Account Information and Data, Integration and Synchronization, Account Features and Customization, Miscellaneous.
- Storage/Memory: Types of Storage, Capacity and Size, Upgrade and Expansion, Performance and Speed, Storage Management, Brands and Models, Technical Specifications, Requirements and Cost, Miscellaneous.

Here is the user prompt: 
"""
# Define sub_aspect_extraction function (existing code)
def sub_aspect_extraction(user_prompt):
    try:
        response = client.completions.create(
            model=deployment_name,
            prompt=start_phrase + user_prompt,
            max_tokens=10000,
            temperature=0
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error processing review: {e}")
        return "Generic"
    
def calculate_net_sentiment(group):
    total_sentiment_score = group['Sentiment_Score'].sum()
    total_review_count = group['Review_Count'].sum()

    if total_review_count == 0:
        return 0
    else:
        return (total_sentiment_score / total_review_count) * 100
        
        
def calculate_negative_review_percentage(group):
    total_reviews = group['Review_Count'].sum()
    negative_reviews = group[group['Sentiment_Score'] == -1]['Review_Count'].sum()
    
    if total_reviews == 0:
        return 0
    else:
        return (negative_reviews / total_reviews) * 100
        
        
def get_conversational_chain_hypothetical_summary():
    global model
    global history
    try:
        prompt_template = """  
        
        As an AI assistant, your task is to provide a detailed summary based on the aspect-wise sentiment and its negative percentages, focusing on the top contributing aspects in terms of review count. Ensure that the analysis explains the percentage of negative reviews for each aspect, highlighting the reasons behind them.
        ## Instructions:
        1. Summarize the reviews for only **top 5** aspects with the highest review counts excluding Generic aspect, clearly stating the **negative percentage** for them.
        2. Explain the **reasons** for the negative feedback for these aspects, focusing on specific points mentioned by users.
        3. Provide a detailed breakdown of the aspects and ensure the negative summary is well-structured, with the main reasons for user **dissatisfaction highlighted** with negative percentages in the below format except for Generic Aspect.
            Format: 1.Aspect1:**Negative_Summary, Negative Percentage**
                    2.Aspect2:**Negative_Summary, Negative Percentage**
                    3.Aspect3:**Negative_Summary, Negative Percentage**
                    4.Aspect4:**Negative_Summary, Negative Percentage**
                    5.Aspect5:**Negative_Summary, Negative Percentage**
                    Conclusion:**    **
                    NOTE:Do not include Generic Aspect in above format, Do not mention more then 5 aspects based on highest review count
                    
        4. End with a brief conclusion summarizing the painpoints of customers in specific aspects which are having more negative percentages, indicating which areas need the most attention.
        5. When delivering the output, be confident and avoid using future tense expressions like "could be" or "may be."
         
        Context:The DataFrame contains aspect-wise review data, including sentiment scores, review counts, and calculated negative review percentages. The summary should reflect these metrics, emphasizing the negative sentiment and its causes in aspects that contribute the most to the overall review count.
        Note:**Mention the negative review percentages for top 5 aspects based on highest review count with negative summary** except Generic aspect
 
        Context:\n {context}?\n
        Question: \n{question}\n
 
        Answer:
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(azure_deployment="Thruxton_R",api_version='2024-03-01-preview',temperature = 0.0)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        return err
# Function to handle user queries using the existing vector store
def hypothetical_summary(user_question, vector_store_path="faiss_index_Copilot+PC Subaspect"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_hypothetical_summary()
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = f"An error occurred while getting LLM response for detailed review summarization: {e}"
        return err
    
def assign_sentiment(row):
    subaspect = row['Sub_Aspect']
    current_sentiment = row.get('Sentiment_Score', 0)

    if pd.isna(subaspect):
        return current_sentiment

    subaspect = subaspect.lower()

    if any(pos.lower() in subaspect for pos in positive_aspects):
        return 1
    elif any(neg.lower() in subaspect for neg in negative_aspects):
        return -1
    else:
        return current_sentiment

device_family_names = df['Device Names'].unique()

# Streamlit sidebar with a dropdown for Device Names selection
selected_device_family = st.sidebar.selectbox("Select Device Family Name:", device_family_names)

# Filter the DataFrame based on the selected Device Family Name
filtered_df_device = df[df['Device Names'] == selected_device_family]

# If a Device Family is selected, show the next dropdown for Processor Generation
if not filtered_df_device.empty:
    st.header("Net Sentiment Forcasting Model")
    processor_generations = filtered_df_device['ProcessorGeneration_Tagged'].unique()
    selected_processor_generation = st.sidebar.selectbox("Select Processor Generation:", processor_generations)

    # Filter the DataFrame based on the selected Processor Generation
    filtered_df_processor = filtered_df_device[filtered_df_device['ProcessorGeneration_Tagged'] == selected_processor_generation]

    # If Processor Generation is selected, show the next dropdown for SS_Tagged
    if not filtered_df_processor.empty:
        ss_options = filtered_df_processor['SS_Tagged'].unique()
        selected_ss = st.sidebar.selectbox("Select Screen Size:", ss_options)

        # Filter the DataFrame based on the selected SS_Tagged
        filtered_df_ss = filtered_df_processor[filtered_df_processor['SS_Tagged'] == selected_ss]

        # If SS_Tagged is selected, show the next dropdown for HardDriveSize_Tagged
        if not filtered_df_ss.empty:
            hard_drive_sizes = filtered_df_ss['HardDriveSize_Tagged'].unique()
            selected_hard_drive_size = st.sidebar.selectbox("Select Hard Drive Size:", hard_drive_sizes)

            # Filter the DataFrame based on the selected HardDriveSize_Tagged
            filtered_df_hard_drive = filtered_df_ss[filtered_df_ss['HardDriveSize_Tagged'] == selected_hard_drive_size]

            # If HardDriveSize_Tagged is selected, show the final dropdown for RAM_Tagged
            if not filtered_df_hard_drive.empty:
                ram_options = filtered_df_hard_drive['RAM_Tagged'].unique()
                selected_ram = st.sidebar.selectbox("Select RAM Size:", ram_options)

                # Filter the DataFrame based on the selected RAM_Tagged
                filtered_df_final = filtered_df_hard_drive[filtered_df_hard_drive['RAM_Tagged'] == selected_ram]
                # filtered_df_final.to_csv("sample_file.csv")

                # Now filtered_df_final contains the filtered data based on all selections
                # The rest of your logic remains the same to display the filtered data and perform the calculations

                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Display chat messages from history on app rerun
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # React to user input
                if user_prompt := st.chat_input("Enter your prompt"):
                    st.chat_message("user").markdown(user_prompt)
                    st.session_state.messages.append({"role": "user", "content": user_prompt})

                    result = sub_aspect_extraction(user_prompt)
                    print(result)

                    positive_match = re.search(r'Positive\s*:\s*(.*?)\s*Negative', result, re.DOTALL)
                    negative_match = re.search(r'Negative\s*:\s*(.*)', result, re.DOTALL)

                    positive_aspects = positive_match.group(1).split(',') if positive_match else []
                    negative_aspects = negative_match.group(1).split(',') if negative_match else []

                    positive_aspects = [aspect.strip() for aspect in positive_aspects]
                    negative_aspects = [aspect.strip() for aspect in negative_aspects]

                    print(f"Positive Aspects: {positive_aspects}")
                    print(f"Negative Aspects: {negative_aspects}")

                    # Aspect-wise net sentiment and review count (your previous logic)
                    col1, col2 = st.columns(2)

                    with col1:
                        net_sentiment = calculate_net_sentiment(filtered_df_final)
                        actual_sentiment_message = f"**Actual Net Sentiment:** {net_sentiment:.2f}%"
                        st.write(actual_sentiment_message)

                        aspect_wise_net_sentiment1 = filtered_df_final.groupby('Aspect').apply(lambda group: pd.Series({
                            'Net_Sentiment': calculate_net_sentiment(group),
                            'Review_Count': group['Review_Count'].sum(),
                            'Negative_Percentage': calculate_negative_review_percentage(group)
                        })).reset_index()

                        aspect_wise_net_sentiment1 = aspect_wise_net_sentiment1[aspect_wise_net_sentiment1['Review_Count'] >= 200]
                        aspect_wise_net_sentiment1 = aspect_wise_net_sentiment1.sort_values(by='Review_Count', ascending=False)
                        aspect_wise_net_sentiment1['Net_Sentiment'] = aspect_wise_net_sentiment1['Net_Sentiment'].apply(lambda x: f"{x:.2f}%")
                        aspect_wise_net_sentiment1['Negative_Percentage'] = aspect_wise_net_sentiment1['Negative_Percentage'].apply(lambda x: f"{x:.2f}%")
                        st.write(aspect_wise_net_sentiment1)

                    with col2:
                        filtered_df_final['Sentiment_Score'] = filtered_df_final.apply(assign_sentiment, axis=1)
                        net_sentiment = calculate_net_sentiment(filtered_df_final)
                        forecasted_sentiment_message = f"**Forecasted Net Sentiment:** {net_sentiment:.2f}%"
                        st.write(forecasted_sentiment_message)

                        aspect_wise_net_sentiment = filtered_df_final.groupby('Aspect').apply(lambda group: pd.Series({
                            'Net_Sentiment': calculate_net_sentiment(group),
                            'Review_Count': group['Review_Count'].sum(),
                            'Negative_Percentage': calculate_negative_review_percentage(group)
                        })).reset_index()

                        aspect_wise_net_sentiment = aspect_wise_net_sentiment[aspect_wise_net_sentiment['Review_Count'] >= 200]
                        aspect_wise_net_sentiment = aspect_wise_net_sentiment.sort_values(by='Review_Count', ascending=False)
                        aspect_wise_net_sentiment['Net_Sentiment'] = aspect_wise_net_sentiment['Net_Sentiment'].apply(lambda x: f"{x:.2f}%")
                        aspect_wise_net_sentiment['Negative_Percentage'] = aspect_wise_net_sentiment['Negative_Percentage'].apply(lambda x: f"{x:.2f}%")
                        st.write(aspect_wise_net_sentiment)

                    forecasted_summary = hypothetical_summary(result + str(aspect_wise_net_sentiment.to_dict()) + "Based on this above data please provide proper summary for the user question: " + user_prompt)
                    st.write(forecasted_summary)

                    st.session_state.messages.append({"role": "assistant", "content": actual_sentiment_message})
                    st.session_state.messages.append({"role": "assistant", "content": forecasted_sentiment_message})
                    st.session_state.messages.append({"role": "assistant", "content": forecasted_summary})

   