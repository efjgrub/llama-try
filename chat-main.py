# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional
import fire
from llama import Llama, Dialog
from flask import Flask, request, jsonify

app = Flask(__name__)

def predict():
    dialogs: List[Dialog] = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json() 
        return jsonify({"sucess": "Lets to predict"}) 
    else:
        return jsonify({"error": "Request body must be JSON"}), 400

# Command line
# torchrun --nproc_per_node 1 chat-main.py 

if __name__ == "__main__":
    ckpt_dir = 'llama-2-7b-chat/'
    tokenizer_path = "tokenizer.model"
    temperature  = 0.6
    top_p = 0.9
    max_seq_len = 512
    max_batch_size = 8
    max_gen_len = None

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    app.run(debug=True)
