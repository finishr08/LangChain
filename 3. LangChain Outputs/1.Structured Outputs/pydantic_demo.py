from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):

    name: str = 'Mustafa'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=4, default=1, description='A decimal value representing the cgpa of the student')


new_student = {'age':'20', 'email':'mustafa@gmail.com', 'cgpa':3.7}

student = Student(**new_student)

student_dict = dict(student)

print(student_dict)

student_json = student.model_dump_json()