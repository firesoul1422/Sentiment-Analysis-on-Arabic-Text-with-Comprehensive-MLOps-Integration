from prefect import task, flow
from prefect.client.schemas.schedules import IntervalSchedule
from datetime import timedelta, datetime
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training import training

@flow(name="Training Flow")
def training_flow():
    train()

@task
def train():
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/Tweets.txt"))
    training(data_path)

# Serve the flow with a schedule
if __name__ == "__main__":
    training_flow.serve(
        name="training-deployment",
        schedules=[
            IntervalSchedule(
                interval=timedelta(weeks=4),  # Run every 4 weeks
                anchor_date=datetime(2025, 1, 18),  # Start date
            )
        ],
        tags=["training"],
    )