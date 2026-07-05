import subprocess
import time
import os
import signal

def main():
    print("Starting Streamlit app in background...")
    streamlit_cmd = [
        r"D:\Computer_Vision\venv\Scripts\streamlit.exe", 
        "run", 
        r"d:\Computer_Vision\LogFormer\app.py", 
        "--server.port", "8999", 
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--client.toolbarMode", "hidden"
    ]
    
    # Start streamlit as a background process with output routed to DEVNULL to prevent deadlock
    proc = subprocess.Popen(streamlit_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for the model loading and similarity database to index (approx 35 seconds)
    print("Waiting 35 seconds for model and vector similarity index to build...")
    time.sleep(35)
    
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    output_path = r"d:\Computer_Vision\LogFormer\plots\logmind_dashboard_preview.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Headless chrome screenshot command with virtual time budget to allow JS to render
    chrome_cmd = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        "--window-size=1280,800",
        "--virtual-time-budget=20000",
        "--hide-scrollbars",
        f"--screenshot={output_path}",
        "http://localhost:8999"
    ]
    
    print("Running Headless Chrome to capture screenshot...")
    try:
        subprocess.run(chrome_cmd, timeout=10)
        if os.path.exists(output_path):
            print(f"Successfully captured real dashboard screenshot at {output_path}!")
        else:
            print("Chrome execution completed but screenshot file was not created.")
    except subprocess.TimeoutExpired:
        print("Chrome capture command timed out.")
    except Exception as e:
        print(f"Error executing Chrome capture: {e}")
        
    print("Terminating Streamlit background server...")
    proc.terminate()
    proc.wait()
    print("Done!")

if __name__ == "__main__":
    main()
