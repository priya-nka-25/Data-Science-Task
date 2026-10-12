# Enterprise Security & Compliance Specifications

## Section 3.1: Authentication, MFA, and SSO Integration
All user accounts require Multi-Factor Authentication (MFA) utilizing Time-based One-Time Passwords (TOTP) or FIDO2 WebAuthn hardware security keys. Enterprise plans include Single Sign-On (SSO) integration supporting SAML 2.0 and OpenID Connect (OIDC) protocols with providers including Okta, Azure Active Directory, PingFederate, and Google Workspace. Just-In-Time (JIT) user provisioning and SCIM 2.0 automated user deprovisioning are fully supported.

## Section 3.2: Password Complexity & Rotation Policy
Passwords must contain a minimum of 14 characters including uppercase letters, lowercase letters, numbers, and symbols. Passwords must not match any of the previous 10 passwords. Enterprise organizations can configure custom mandatory password expiration intervals (default: 90 days). Accounts are automatically locked after 5 consecutive failed login attempts for a duration of 30 minutes unless unlocked by an administrator.

## Section 3.3: Data Encryption Standards and SOC 2 Compliance
Data in transit is encrypted using TLS 1.3 with AES-256-GCM cipher suites. Data at rest is encrypted using FIPS 140-3 validated AES-256 encryption with customer-managed keys (BYOK) supported in enterprise tiers. The platform undergoes annual SOC 2 Type II audits, ISO 27001 certification renewals, and third-party penetration tests every six months.

