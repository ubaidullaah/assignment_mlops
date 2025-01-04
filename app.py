import gradio as gr
from model import train_model

# Train the model
model = train_model()

# Prediction function that the Gradio interface will use
def predict_score(hours_studied, attendance_rate):
    prediction = model.predict([[hours_studied, attendance_rate]])
    return f"Predicted Final Score: {prediction[0]:.2f}"

# Setup Gradio interface
iface = gr.Interface(fn=predict_score,
                     inputs=[gr.Number(label="Hours Studied"),
                             gr.Slider(minimum=0, maximum=100, step=1, label="Attendance Rate")],
                     outputs="text",
                     title="Student Score Prediction",
                     description="Enter the number of hours studied and attendance rate to predict the final score.")

# Run the application
if __name__ == "__main__":
    iface.launch()
