import pickle

with open("models/product_category_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Unesi naziv proizvoda (ili 'exit')")

while True:
    title = input("> ")
    if title.lower() == "exit":
        break

    prediction = model.predict([title])
    print("Predlozena kategorija:", prediction[0])
