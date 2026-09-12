from src.preprocessing.metadata_readers.asllrp_reader import ASLLRPReader

reader = ASLLRPReader(xml_dir="data/raw/asllrp/xml", video_dir="data/raw/asllrp/videos")
utt = next(iter(reader))
print(f"utterance_id: {utt.utterance_id}")
print(f"num signs: {len(utt.glosses)}")
print(f"glosses: {utt.glosses}")
print(f"sign_spans: {utt.sign_spans}")
print(f"num spans found (non-None): {sum(1 for s in utt.sign_spans if s is not None)}")