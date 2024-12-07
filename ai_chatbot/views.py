from django.http import JsonResponse
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Naplózó beállítása
logger = logging.getLogger(__name__)

# Modell és Tokenizer betöltése
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def generate_response(question):
    try:
        # Pad token beállítása
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer.encode(question, return_tensors="pt", padding=True, truncation=True)
        # attention_mask és pad_token_id beállítása
        attention_mask = inputs.ne(tokenizer.pad_token_id).float()
        # Modell válasz generálása
        outputs = model.generate(inputs, max_length=100, num_return_sequences=1, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    except Exception as e:
        logger.error(f"Generálási hiba: {str(e)}")
        return None

def chatbot_response(request):
    if request.method == "POST":
        question = request.POST.get("question")  # Kérdés a frontendről
        if not question:
            logger.error("Nincs kérdés megadva.")
            return JsonResponse({"error": "No question provided"}, status=400)

        logger.info(f"Kérdés érkezett: {question}")

        try:
            # Válasz generálása
            answer = generate_response(question)
            if not answer:
                logger.error("A válasz generálása nem sikerült.")
                return JsonResponse({"error": "Hiba történt a válasz generálása közben."}, status=500)

            logger.info(f"Generált válasz: {answer}")
            return JsonResponse({"reply": answer})
        except Exception as e:
            logger.error(f"Hiba történt a válasz küldése közben: {str(e)}")
            return JsonResponse({"error": f"Hiba történt a válasz küldése közben: {str(e)}"}, status=500)
    else:
        logger.error("Érvénytelen kérés: nem POST típus.")
        return JsonResponse({"error": "Invalid request method"}, status=400)