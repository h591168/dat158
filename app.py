import gradio as gr
import pandas as pd
import joblib

# Load your trained model
model = joblib.load('rf_model.joblib')

# Define a placeholder value for 'collection_id' if it's not part of user input
# This could be the median or mode of 'collection_id' from the training dataset, or another appropriate placeholder value
placeholder_collection_id = 0

# Function to process input and make predictions
def predict_revenue(budget, popularity, runtime):
    # Create a DataFrame with the input values
    input_data = pd.DataFrame([[budget, popularity, runtime, placeholder_collection_id]],
                              columns=['budget', 'popularity', 'runtime', 'collection_id'])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    return f'The predicted revenue is ${prediction[0]:,.2f}'

# Define the Gradio interface
interface = gr.Interface(
    fn=predict_revenue,
    inputs=[
        gr.inputs.Number(label="Enter the budget"),
        gr.inputs.Number(label="Enter the popularity"),
        gr.inputs.Number(label="Enter the runtime")
    ],
    outputs="text",
    title="Revenue Prediction Model",
    description="Enter the budget, popularity, and runtime to predict the box office revenue."
)

# Launch the app
if __name__ == "__main__":
  interface.launch()
 

