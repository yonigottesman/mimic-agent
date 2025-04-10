+++
date = "2015-09-01T19:34:46-04:00"
title = "D_ICD_PROCEDURES"
linktitle = "D_ICD_PROCEDURES"
weight = 17
toc = "true"

+++

**Table source:** Online sources.

**Table purpose:** Definition table for ICD procedures.

**Number of rows:** 3,882

**Links to:**

* PROCEDURES_ICD on `ICD9_CODE`

# Brief summary

This table defines International Classification of Diseases Version 9 (ICD-9) codes for **procedures**. These codes are assigned at the end of the patient's stay and are used by the hospital to bill for care provided. They can further be used to identify if certain procedures have been performed (e.g. surgery).

<!-- # Important considerations -->

# Table columns

Name | Postgres data type
---- | ----
ROW\_ID | INT
ICD9\_CODE | VARCHAR(10)
SHORT\_TITLE | VARCHAR(50)
LONG\_TITLE | VARCHAR(300)

# Detailed Description

## `ICD9_CODE`

`ICD9_CODE` is the International Coding Definitions Version 9 (ICD-9) code. Each code corresponds to a single procedural concept.

## `SHORT_TITLE`, `LONG_TITLE`

The title fields provide a brief definition for the given procedure code in `ICD9_CODE`.
