from pydantic import BaseModel,Field
from typing import Annotated

class UserInput(BaseModel):

    Age:Annotated[float,Field(...,description="Enter the Age:",examples=[21])]
    Attendance:Annotated[float,Field(...,description="Enter the Attendance:",examples=[80])]
    StudyHours:Annotated[float,Field(...,description="Enter the StudyHours:",examples=[40])]
    PreviousMarks:Annotated[float,Field(...,description="Enter the PreviousMarks:",examples=[21])]
    SleepHours:Annotated[float,Field(...,description="Enter the SleepHours:",examples=[8])]
    Semester1:Annotated[float,Field(...,description="Enter the Semester1:",examples=[84])]
    Semester2:Annotated[float,Field(...,description="Enter the Semester2:",examples=[74])]
    Semester3:Annotated[float,Field(...,description="Enter the Semester3:",examples=[69])]
    Semester4:Annotated[float,Field(...,description="Enter the Semester4:",examples=[90])]