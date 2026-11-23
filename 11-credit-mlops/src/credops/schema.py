"""Column lists and the request/response schema shared by training and serving.

Keeping the schema in one module means the API cannot silently accept a feature the model never saw.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

TARGET = "default"
ID_COL = "ID"
PAY_STATUS = [f"PAY_{i}" for i in range(1, 7)]
BILL_AMT = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]
PROTECTED = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]
MODEL_INPUTS = ["LIMIT_BAL", *PAY_STATUS, *BILL_AMT, *PAY_AMT]
"""Raw columns the model uses. Protected attributes are never model inputs (see README, ethics)."""

PayStatus = Field(ge=-2, le=9, description="Repayment status: -2 no use, -1 paid duly, 0 revolving, k = k months late")
Bill = Field(ge=-1_000_000, le=10_000_000, description="Statement amount (NT$); negative means a credit balance")
Paid = Field(ge=0, le=10_000_000, description="Amount paid (NT$)")


class Applicant(BaseModel):
    """One client's last six months. Month 1 is the most recent (September 2005 in the source data)."""

    model_config = ConfigDict(extra="forbid")

    LIMIT_BAL: int = Field(gt=0, le=10_000_000, description="Credit limit (NT$)")
    PAY_1: int = PayStatus
    PAY_2: int = PayStatus
    PAY_3: int = PayStatus
    PAY_4: int = PayStatus
    PAY_5: int = PayStatus
    PAY_6: int = PayStatus
    BILL_AMT1: int = Bill
    BILL_AMT2: int = Bill
    BILL_AMT3: int = Bill
    BILL_AMT4: int = Bill
    BILL_AMT5: int = Bill
    BILL_AMT6: int = Bill
    PAY_AMT1: int = Paid
    PAY_AMT2: int = Paid
    PAY_AMT3: int = Paid
    PAY_AMT4: int = Paid
    PAY_AMT5: int = Paid
    PAY_AMT6: int = Paid


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[Applicant] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    request_id: str
    probability: float = Field(description="Calibrated P(default next month)")
    flag: bool = Field(description="probability >= decision threshold (a referral flag for a human reviewer)")
    threshold: float
    model_version: str


class BatchPrediction(BaseModel):
    request_id: str
    probabilities: list[float]
    flags: list[bool]
    threshold: float
    model_version: str
