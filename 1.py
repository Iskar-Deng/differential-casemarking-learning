from transformers import GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("/workspace/differential-casemarking-learning/checkpoints/independent_Anone_Pdefinite/checkpoint-1000")
print("🄿" in tok.get_vocab())
print(tok.tokenize("🄿"))

 wc -l /workspace/differential-casemarking-learning/data/perturbed/independent_Anone_Pdefinite/train_affected.txt \
    /workspace/differential-casemarking-learning/data/perturbed/independent_Anone_Pdefinite/train_unaffected.txt \
      /workspace/differential-casemarking-learning/data/perturbed/independent_Anone_Pdefinite/train_invalid.txt

 wc -l /workspace/differential-casemarking-learning/data/perturbed/independent_Anone_Ppronoun/train_affected.txt \
    /workspace/differential-casemarking-learning/data/perturbed/independent_Anone_Ppronoun/train_unaffected.txt \
      /workspace/differential-casemarking-learning/data/perturbed/independent_Anone_Ppronoun/train_invalid.txt