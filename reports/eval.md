# Retrieval Evaluation

| Mode | Set | Questions | Recall@3 | Key-fact rate | Refusal accuracy | Wrong-answer rate |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | dev (25 answerable, 5 refusals) | 30 | 100.0% | 100.0% | 100.0% | 0.0% |
| bm25 | held-out (12 answerable, 3 refusals) | 15 | 58.3% | 33.3% | 33.3% | 26.7% |
| hybrid | dev (25 answerable, 5 refusals) | 30 | 100.0% | 100.0% | 100.0% | 0.0% |
| hybrid | held-out (12 answerable, 3 refusals) | 15 | 83.3% | 33.3% | 100.0% | 6.7% |

Wrong-answer rate is answered-but-incorrect questions (including answered refusals) divided by all questions in that set. Dev alone may guide thresholds; held-out questions and scores are reported without tuning. Hybrid thresholds (RRF >= 0.025; query-to-answer cosine >= 0.45) were selected using dev only.

## Held-out incorrect responses

### bm25

#### h01: If my tent fails, how long is the guarantee?

- System answer: I don't know based on the available documents.
- Expected answer: 2-year limited warranty
- Reason: Answer omitted expected facts: 2-year, limited warranty.

#### h02: Can I send back something unused, and by when?

- System answer: Up to 5 unused vacation days may carry into the next year and must be used by March 31.
Source: leave.md — Carryover and sick leave
- Expected answer: Within 30 days after delivery
- Reason: Answer omitted expected facts: 30 days, delivery.

#### h03: My parcel arrived broken. How quickly do I need to tell you?

- System answer: I don't know based on the available documents.
- Expected answer: Within 7 days
- Reason: Answer omitted expected facts: 7 days, damaged.

#### h05: When is regular delivery on the house?

- System answer: I don't know based on the available documents.
- Expected answer: Orders of $75 or more before tax
- Reason: Answer omitted expected facts: $75, free.

#### h06: How many days does ground shipping take once it leaves?

- System answer: Alaska, Hawaii, and PO boxes are excluded from the 4–7 business day estimate; those destinations usually take 7–12 business days.
Source: shipping.md — Expedited and remote destinations
- Expected answer: 4–7 business days after dispatch
- Reason: Answer omitted expected facts: 4–7 business days, dispatch.

#### h08: What can I take off my bill with a hundred points?

- System answer: I don't know based on the available documents.
- Expected answer: $5 discount
- Reason: Answer omitted expected facts: 100 points, $5.

#### h11: How long do I have to turn in an expense claim?

- System answer: I don't know based on the available documents.
- Expected answer: Within 30 days with receipt
- Reason: Answer omitted expected facts: 30 days, receipt.

#### h12: What's the per-mile rate for work trips?

- System answer: I don't know based on the available documents.
- Expected answer: $0.67 per mile
- Reason: Answer omitted expected facts: $0.67, per mile.

#### h14: Does Acme say what its chief executive's favourite colour is?

- System answer: Acme posts active recalls on its website and contacts registered purchasers by email when contact details are available.
Source: safety.md — Recall notices
- Expected answer: None
- Reason: Answered an unanswerable question.

#### h15: Which factory makes the backpacks sold by Acme?

- System answer: Customers may request a copy or correction of account information by contacting privacy@acme.example; Acme responds within 30 days after verifying the requester's identity.
Source: privacy.md — Data retention and access
- Expected answer: None
- Reason: Answered an unanswerable question.

### hybrid

#### h01: If my tent fails, how long is the guarantee?

- System answer: I don't know based on the available documents.
- Expected answer: 2-year limited warranty
- Reason: Answer omitted expected facts: 2-year, limited warranty.

#### h02: Can I send back something unused, and by when?

- System answer: I don't know based on the available documents.
- Expected answer: Within 30 days after delivery
- Reason: Answer omitted expected facts: 30 days, delivery.

#### h03: My parcel arrived broken. How quickly do I need to tell you?

- System answer: I don't know based on the available documents.
- Expected answer: Within 7 days
- Reason: Answer omitted expected facts: 7 days, damaged.

#### h05: When is regular delivery on the house?

- System answer: I don't know based on the available documents.
- Expected answer: Orders of $75 or more before tax
- Reason: Answer omitted expected facts: $75, free.

#### h06: How many days does ground shipping take once it leaves?

- System answer: Alaska, Hawaii, and PO boxes are excluded from the 4–7 business day estimate; those destinations usually take 7–12 business days.
Source: shipping.md — Expedited and remote destinations
- Expected answer: 4–7 business days after dispatch
- Reason: Answer omitted expected facts: 4–7 business days, dispatch.

#### h08: What can I take off my bill with a hundred points?

- System answer: I don't know based on the available documents.
- Expected answer: $5 discount
- Reason: Answer omitted expected facts: 100 points, $5.

#### h11: How long do I have to turn in an expense claim?

- System answer: I don't know based on the available documents.
- Expected answer: Within 30 days with receipt
- Reason: Answer omitted expected facts: 30 days, receipt.

#### h12: What's the per-mile rate for work trips?

- System answer: I don't know based on the available documents.
- Expected answer: $0.67 per mile
- Reason: Answer omitted expected facts: $0.67, per mile.
