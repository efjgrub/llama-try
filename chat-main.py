from typing import List, Optional
import fire
from llama import Llama, Dialog
from flask import Flask, jsonify, request

app = Flask(__name__)

def predict():
    dialogs: List[Dialog] = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=2048,
        temperature=0.3,
        top_p=0.9,
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
    
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    app.run(debug=True)
    

if __name__ == "__main__":
    global generator
    fire.Fire(main)

