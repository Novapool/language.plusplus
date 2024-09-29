import os
import subprocess
import json

def get_file_format(file_path):
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', file_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error probing {file_path}: {result.stderr}")
        return None
    try:
        probe_data = json.loads(result.stdout)
        return probe_data.get('format', {}).get('format_name', '')
    except json.JSONDecodeError:
        print(f"Error parsing probe data for {file_path}")
        return None

def convert_opus_to_wav(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".opus"):
                opus_path = os.path.join(root, file)
                relative_path = os.path.relpath(opus_path, input_folder)
                wav_path = os.path.join(output_folder, os.path.splitext(relative_path)[0] + ".wav")
                
                os.makedirs(os.path.dirname(wav_path), exist_ok=True)
                
                file_format = get_file_format(opus_path)
                
                try:
                    if file_format and 'ogg' in file_format:
                        # File is in Ogg container
                        cmd = ['ffmpeg', '-i', opus_path, wav_path]
                    else:
                        # Assume raw Opus data
                        cmd = ['ffmpeg', '-f', 'opus', '-i', opus_path, wav_path]
                    
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    print(f"Converted {opus_path} to {wav_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Error converting {opus_path}: {e}")
                    print(f"FFmpeg output: {e.stdout}")
                    print(f"FFmpeg error: {e.stderr}")

if __name__ == "__main__":
    input_folder = "dataset"
    output_folder = "dataset_wav"
    convert_opus_to_wav(input_folder, output_folder)