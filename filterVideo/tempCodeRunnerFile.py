 os.path.exists(video_path) :
                    process_video(video_path, upsample=upsample, format=format, fps=fps )
                else:
                    print(f"❌ Error: Video file '{video_path}' does not exist.")
                    files = [f for f in os.listdir(".") if os.path.isfile(f)]
                    