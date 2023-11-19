import os
import subprocess

def convert_video_to_audio(input_video, output_audio):
    command = f"ffmpeg -i {input_video} -ar 44100 {output_audio}"
    subprocess.run(command, shell=True)

def convert_videos_in_folder(input_folder, output_folder):
    for ep_name in os.listdir(input_folder):
        in_ep_folder = input_folder+"/"+ep_name
        out_ep_folder = output_folder+"/"+ep_name
        if not os.path.exists(out_ep_folder):
            os.makedirs(out_ep_folder)
        
        for file_name in os.listdir(in_ep_folder):
            command = f"ffmpeg -i {os.path.join(in_ep_folder,file_name)} -ar 44100 {os.path.join(out_ep_folder,file_name)}"
            subprocess.run(command, shell=True)
# HYPERPARAMS TO MODIFY
season = "S10"
# END HYPERPARAMS TO MODIFY
input_folder = "Clips_from_Script/"+season
output_folder = "Clips_from_Script_correct_rate/"+season
convert_videos_in_folder(input_folder, output_folder)