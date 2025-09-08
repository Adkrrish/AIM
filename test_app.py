import streamlit as st

def main():
    st.title("🔧 Streamlit Test - Working!")
    st.write("If you see this, Streamlit is rendering correctly.")
    st.success("✅ App is functional")
    
    # Basic functionality test
    if st.button("Test Button"):
        st.balloons()
        st.write("Button clicked successfully!")

if __name__ == '__main__':
    main()
