+++
date = "2015-09-01T19:34:46-04:00"
title = "The caregivers table"
linktitle = "CAREGIVERS"
weight = 12
toc = "true"

+++

**Table source:** CareVue and Metavision ICU databases.

**Table purpose:** Defines the role of caregivers.

**Number of rows:** 7567

**Links to:**

* CHARTEVENTS on `CGID`

# Brief summary

This table provides information regarding care givers. For example, it would define if a care giver is a research nurse (RN), medical doctor (MD), and so on.

<!-- # Important considerations -->

# Table columns

Name | Postgres data type
---- | ----
ROW\_ID | INT
CGID | INT
LABEL | VARCHAR(15)
DESCRIPTION | VARCHAR(30)

# Detailed Description

The CAREGIVERS table provides information regarding the type of caregiver. Each caregiver is represented by a unique integer which maps to this table.

## `CGID`

`CGID` is a unique identifier for each distinct caregiver present in the database. `CGID` is sourced from two tables in the raw data: the CareVue and Metavision ICU databases. Due to imprecision in the storage of unique identifiers across the database, it is possible that two distinct caregivers with the same names (e.g. RN Sarah Jones and MD Sarah Jones) would be considered as the same caregiver. However, this is an unlikely occurrence.

## `LABEL`

`LABEL` defines the type of caregiver: e.g. RN, MD, PharmD, etc. Note that `LABEL` is a free text field and as such contains many typographical errors and spelling variants of the same concept (e.g. MD, MDs, M.D.).

## `DESCRIPTION`

`DESCRIPTION` is less frequently present than `LABEL`, and provides additional information regarding the caregiver. This column is much more structured, and contains only 17 unique values as of MIMIC-III v1.0.
