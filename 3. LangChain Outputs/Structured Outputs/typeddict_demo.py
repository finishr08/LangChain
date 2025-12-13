from typing import TypedDict

class Person(TypedDict):

    name: str
    age: int

new_person: Person = {'name':'Mustafa Ahmed', 'age':20} # 20 in 2025

print(new_person)