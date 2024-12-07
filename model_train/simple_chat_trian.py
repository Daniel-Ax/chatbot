from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Betöltés
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")

# Adatok betöltése
dataset = load_dataset("squad_v2")

# Tokenizálás előfeldolgozó függvény
def preprocess_function(examples):
    # Tokenizálás
    tokenized_examples = tokenizer(examples['question'], examples['context'], truncation=True, padding="max_length")
    
    # Inicializáljuk a kezdő és vég pozíciók listáit
    start_positions = []
    end_positions = []
    
    for i in range(len(examples['answers'])):
        answer = examples['answers'][i]  # Válasz elem
        # Ellenőrizzük, hogy a válasz létezik-e, és van-e kezdő pozíció
        if 'answer_start' in answer and len(answer['answer_start']) > 0:
            start_positions.append(answer['answer_start'][0])  # Válasz kezdete
            end_positions.append(answer['answer_start'][0] + len(answer['text'][0]))  # Válasz vége
        else:
            start_positions.append(-1)  # Ha nincs válasz, akkor -1
            end_positions.append(-1)    # Ha nincs válasz, akkor -1
    
    tokenized_examples['start_positions'] = start_positions
    tokenized_examples['end_positions'] = end_positions
    
    return tokenized_examples

# Tokenizálás alkalmazása az adathalmazra
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Ellenőrzés, hogy a tokenizálás rendben zajlott
print(tokenized_datasets['train'][0])  # Ellenőrizd az első példát a tanuló adathalmazban

# Képzési beállítások
training_args = TrainingArguments(
    output_dir="./results",          # Eredmények mappája
    num_train_epochs=3,              # Epochok száma
    per_device_train_batch_size=8,   # Batch size
    per_device_eval_batch_size=8,    # Evaluation batch size
    warmup_steps=500,                # Lépcsőzetes tanulás
    weight_decay=0.01,               # Súlycsökkentés
    logging_dir="./logs",            # Log fájlok helye
)

# Trainer létrehozása
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"]
)

# Modell tréning
trainer.train()
# Modell mentése
model.save_pretrained("./chatbot_model")
tokenizer.save_pretrained("./chatbot_model")
