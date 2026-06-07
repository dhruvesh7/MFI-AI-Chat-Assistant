# Manual Testing Queries for MFI Chat Assistant

A curated set of validated queries covering all core capabilities of the AI assistant.

> **Note on policy queries (Q10–Q13):** These require the server to be restarted after running `python ingest.py` so the updated knowledge base is loaded into memory.

---

### General Company Information
1. What does Money Forward India do?
2. Can you tell me about the company's mission and vision?
3. What are the core values of the company?
4. What is the culture like at Money Forward India?
5. What technologies or tech stack does the company use?

### Live Job Listings & Careers
6. What jobs are currently open right now?
7. Do you have any Software Engineer or Developer roles available?
8. I'm looking for an internship. Are there any open positions?

### Policies & Data Security
9. How does Money Forward India handle my personal data?
10. Can you summarize your privacy policy?
11. What security practices and compliance measures does MFI follow?
12. Explain your data protection policy regarding third parties.

### Hiring Process
13. What is the hiring process like at Money Forward India?
14. How many interview rounds are there?

### Multi-part & Edge Cases
15. Can you give me a summary of the company's mission AND list the top 3 open jobs?
16. Can I order a pizza through you? *(Out-of-scope — expected: polite refusal)*
