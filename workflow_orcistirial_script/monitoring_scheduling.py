from prefect import task, flow
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from monitoring import monitoring_calculation
from prefect.client.schemas.schedules import IntervalSchedule
from datetime import timedelta, datetime  



@flow(name="Monitoring Flow")
def Monitoring_flow():
    monitoring()


@task
def monitoring():
    monitoring_calculation()



if __name__ == "__main__":
    Monitoring_flow.serve(
        name="flowing",
        schedules=[
            IntervalSchedule(
                interval=timedelta(weeks=2),  
                anchor_date=datetime(2025, 2, 2), 
            )
        ]
    )