'''
aws_polly(speech generation).py
'''


import boto3

REGION_NAME = 'eu-west-1'

role_session = boto3.session.Session(region_name=REGION_NAME)
polly_client = role_session.client('polly')

words_to_speak = "Hi Ivan, thanks for taking the Hyperclass on Comparing options. You used a total of 355 different words -including 69 difficult ones- and hit 71.43% of the target vocabulary. For example, you used "as" in expressions such as "um then then uh Singapore and uh London and uh there are uh more a sport activity that you could do. Like uh swimming, surfing, jungle, wards, seeing animals and uh the and then and the food uh isn't uh expensive er as ah as uh uh".You also correctly used "because" in the sentence "Mhm Yeah but at the moment is a relief in color to take the ticket uh the the ticket uh uh because mm mm Yes exactly exactly exactly I don't I don't yes the 50% of the place uh".Your longest term was 56 seconds and you spoke for a total of 00 hours 24 mins 22 seconds. Well done!Do you feel like practising some sentences with "For me, or is not as"?"
print(words_to_speak)

voice_id = "Joanna"
language_code = "en-US"
engine = "neural"


response = polly_client.synthesize_speech(
        Text=words_to_speak,
        OutputFormat='mp3',
        VoiceId=voice_id,
        LanguageCode=language_code,
        Engine=engine)

if "AudioStream" in response:
    with closing(response["AudioStream"]) as stream:
        # output = os.path.join("/tmp/", "speech.mp3")

        try:
            with open('audios/tempfile.mp3', 'wb') as f:
                f.write(stream.read())
            temp_aud_file = gr.File("audios/tempfile.mp3")
            temp_aud_file_url = "/file=" + temp_aud_file.value['name']
            html_audio = f'<audio autoplay><source src={temp_aud_file_url} type="audio/mp3"></audio>'
        except IOError as error:
            # Could not write to file, exit gracefully
            print(error)
            return None, None
else:
    # The response didn't contain audio data, exit gracefully
    print("Could not stream audio")
    return None, None