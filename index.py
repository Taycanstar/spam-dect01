from tensorflow.keras.models import load_model
import tensorflow as tf

import tensorflow_hub as hub

# Define custom objects for any TensorFlow Hub layers used
custom_objects = {'KerasLayer': hub.KerasLayer}

# Load the model with custom objects
model = load_model('my_model.h5', custom_objects=custom_objects)

new_emails = [
    "Free money offer!", 
    "Reminder: Meeting at 11 AM", 
    "Special discount for you",
    "Congratulations! You've won a free iPhone! Click here to claim now!",
    "Urgent: You have won $10,000! Send your bank details to receive the cash.",
    "Hot singles in your area waiting to meet you! Don't miss out, join now!",
    "You're selected for a luxury cruise to the Bahamas! Exclusive deal just for you!",
    "Weekend Sale: Get 50% off on all products. Visit our store or shop online.",
    "Special offer just for you: 30% discount on your next flight booking.",
    "Reminder: Your subscription is about to end. Renew now to enjoy uninterrupted service.",
    "We've noticed unusual activity in your account. Click here to verify your details.",
    "Your account will be suspended! Update your password immediately by clicking here.",
    "Hi John, can we reschedule tomorrow's meeting to 2 PM? Let me know your availability.",
    "Reminder: Team lunch this Friday at the downtown restaurant. Please confirm your attendance.",
    "Attached is the report for last quarter's performance. Let's discuss this in our next meeting.",
    "Could you please send me the presentation slides from yesterday's conference?",
    "Hi Mom, just checking in. Hope everything is well. Let's catch up over the weekend.",
    "Planning a small get-together this Saturday evening. Would love for you to come!"
]

# Convert to a TensorFlow dataset (if it's not already in this format)
# and batch it. The preprocessing steps should be the same as your training data
new_data = tf.data.Dataset.from_tensor_slices(new_emails)
new_data = new_data.batch(1)  # Using a batch size of 1 for individual predictions

# Predict
new_predictions = model.predict(new_data)

# If your model outputs logits, convert these to probabilities
new_predictions = tf.sigmoid(new_predictions)

# Convert probabilities to binary class labels (0 or 1)
new_predictions_binary = tf.where(new_predictions < 0.5, 0, 1)

# Display predictions
for email, prediction in zip(new_emails, new_predictions_binary):
    print(f"Email: '{email}' - Predicted: {'Spam' if prediction.numpy()[0] else 'Not Spam'}")

