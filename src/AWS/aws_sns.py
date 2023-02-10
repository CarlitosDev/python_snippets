'''
	AWS Simple Notification System



* A publisher sends messages to topics that they have created or to topics they have permission to publish to. 
* Instead of including a specific destination address in each message, a publisher sends a message to the topic. 
* Amazon SNS matches the topic to a list of subscribers who have subscribed to that topic, and delivers the message to each of those subscribers. 
* Each topic has a unique name that identifies the Amazon SNS endpoint for publishers to post messages and subscribers to register for notifications. 
* Subscribers receive all messages published to the topics to which they subscribe, and all subscribers to a topic receive the same messages.


'''

import boto3

sns = boto3.client('sns')

# Create a topic
topic_name = 'new-teacher'
response = sns.create_topic(Name=topic_name)

print(response)



# Publish a simple message to the specified SNS topic
response = sns.publish(
    TopicArn='arn:aws:sns:region:0123456789:my-topic-arn',    
    Message='Hello World!',    
)



