# Data Retention, Privacy & Compliance Policy

## Section 5.1: GDPR and CCPA Data Subject Access Requests (DSAR)
In compliance with GDPR (Article 15-20) and CCPA, customers can submit automated Data Subject Access Requests (DSAR) via the administration portal or by contacting privacy@company.com. Upon request verification, a machine-readable JSON/CSV export of all personal data is generated within 14 business days. Right-to-be-Forgotten requests trigger an irreversible cryptographic wipe across primary production stores within 30 days.

## Section 5.2: Automated Data Archival and Purge Schedules
Operational logs and audit trails are retained for 365 days in encrypted immutable cold storage before automated permanent erasure. Customer deleted items remain in a soft-deleted recovery staging area for 30 days, after which they are permanently expunged. Disaster recovery database snapshots are rotated and purged on a rolling 90-day retention schedule.

