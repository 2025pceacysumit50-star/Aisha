#!/usr/bin/env python3
"""Test the TTS endpoint"""

import requests
import time

# Test the TTS endpoint
text = 'Hello world. This is AISHA speaking.'
print(f'Testing TTS with text: {text}')

start = time.time()
try:
    response = requests.get(
        'http://localhost:5000/api/tts',
        params={'text': text},
        timeout=60
    )
    elapsed = time.time() - start
    
    print(f'Status: {response.status_code}')
    print(f'Time: {elapsed:.1f}s')
    
    if response.status_code == 200:
        print(f'Content-Type: {response.headers.get("content-type")}')
        print(f'Audio size: {len(response.content)} bytes')
        
        # Save for manual testing
        with open('test_tts_output.wav', 'wb') as f:
            f.write(response.content)
        print('Saved to test_tts_output.wav')
    else:
        print(f'Error: {response.text}')
        
except requests.Timeout:
    print('Request timed out')
except Exception as e:
    print(f'Exception: {e}')
