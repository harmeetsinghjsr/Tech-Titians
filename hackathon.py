import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

class SpamDetectorApp:
    def __init__(self, master):  # Corrected the initialization method name
        self.master = master
        master.title("Spam Detector")

        self.label = tk.Label(master, text="Enter an email to check:")
        self.label.pack()

        self.entry = tk.Entry(master, width=50)
        self.entry.pack()

        self.check_button = tk.Button(master, text="Check", command=self.check_spam)
        self.check_button.pack()

    def check_spam(self):
        email_text = self.entry.get()
        if email_text.strip() == "":
            messagebox.showinfo("Error", "Please enter an email!")
            return

        # Load your trained model
        model = self.load_model()

        # Predict
        prediction = model.predict([email_text])
        
        if prediction[0] == 1:
            messagebox.showinfo("Result", "Spam!")
        else:
            messagebox.showinfo("Result", "Not Spam!")

    def load_model(self):
        # Load your trained model here
        # For this example, I'm just creating a dummy model
        model = make_pipeline(CountVectorizer(), MultinomialNB())
        return model

def main():
    root = tk.Tk()
    app = SpamDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":  # Corrected the main method name
    main()