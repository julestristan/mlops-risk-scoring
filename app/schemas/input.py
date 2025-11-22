# app/schemas/input.py
from pydantic import BaseModel, Field

# Les champs doivent correspondre aux colonnes du DataFrame X utilisé dans train.py
class CreditFeatures(BaseModel):
    
    # Numériques
    Age: int = Field(..., ge=18)
    Job: int = Field(..., ge=0, le=3) 
    Credit_amount: int = Field(..., ge=250)
    Duration: int = Field(..., ge=4)
    
    # Catégorielles (chaînes de caractères brutes)
    Housing: str = Field(..., description="Ex: 'own', 'rent', 'free'")
    Purpose: str = Field(..., description="Ex: 'car', 'education'")
    Sex: str = Field(..., description="Ex: 'male', 'female'") 
    checking_saving_accounts: str # Le feature engineering que vous avez créé

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 45,
                "Job": 2,
                "Credit_amount": 1842,
                "Duration": 24,
                "Housing": "own",
                "Purpose": "furniture/equipment",
                "Sex": "male",
                "checking_saving_accounts": "rich"
            }
        }