import streamlit as st

def main():
    """Minimal application to show installation status and instructions."""
    
    st.set_page_config(
        page_title="Voice Cloner Pro - YourTTS",
        page_icon="🎙️",
        layout="wide",
    )
    
    # Display header
    st.title("🎙️ Advanced Voice Cloner")
    st.subheader("High Quality Voice Cloning with YourTTS")
    
    # Display deployment status
    st.warning("""
    ### Deployment in Progress
    
    The Voice Cloner application is currently being deployed to Streamlit Cloud.
    
    Please check back later or run the application locally for full functionality.
    """)
    
    # Installation instructions
    with st.expander("Local Installation Instructions", expanded=True):
        st.markdown("""
        ### How to Run Locally
        
        For the best experience, run this application locally:
        
        1. Clone the repository:
           ```
           git clone https://github.com/premsingh23/Voice-Cloner--Live-LLM-using-GPT.git
           ```
           
        2. Navigate to the directory:
           ```
           cd Voice-Cloner--Live-LLM-using-GPT
           ```
           
        3. Install dependencies:
           ```
           pip install -r requirements.txt
           ```
           
        4. Run the application:
           ```
           streamlit run app.py
           ```
           
        5. For optimal performance, use a machine with GPU support.
        """)
    
    # GitHub link
    st.markdown("""
    <div style='text-align: center; margin-top: 30px;'>
        <a href="https://github.com/premsingh23/Voice-Cloner--Live-LLM-using-GPT" target="_blank">
            <img src="https://img.shields.io/github/stars/premsingh23/Voice-Cloner--Live-LLM-using-GPT?style=social" alt="GitHub Repo">
            <br>
            View on GitHub
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px; margin-top: 50px;'>
        Built with ❤️ by Principia Team
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 