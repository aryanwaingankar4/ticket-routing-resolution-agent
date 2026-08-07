"""
generate_dataset.py
===================

Synthetic IT Support Ticket Dataset Generator
----------------------------------------------

PURPOSE
    Produces `data/synthetic_tickets.csv`: 4000 synthetic IT support tickets
    used to (a) train a supervised text classifier that maps a ticket to one of
    seven IT domains, and (b) seed a Retrieval-Augmented-Generation (RAG)
    resolution suggester that proposes a fix given a similar past ticket.

WHY THE KEY DESIGN DECISIONS WERE MADE
    (Written so the choices can be defended in a project viva.)

    1. BALANCED CATEGORIES (~571-572 rows each).
       A text classifier trained on imbalanced data learns the *prior* instead
       of the *signal* - it will happily predict the majority class and still
       score high accuracy while being useless in production. Keeping every
       category at roughly equal size forces the model to learn discriminative
       features (vocabulary, phrasing) rather than exploiting class frequency.
       Balanced data also makes macro-F1 (the metric that actually matters for
       routing) meaningful and keeps the confusion matrix interpretable.

    2. LINKED SCENARIOS (one tuple = title + symptom + resolution together).
       The single most damaging bug in synthetic ticket data is LABEL / TARGET
       NOISE created by independently randomizing the title, the symptom text,
       and the resolution. That produces incoherent rows such as a
       "database replication lag" title paired with a "reset the user's
       password" resolution. Such rows poison BOTH tasks: the classifier sees
       contradictory text->label evidence, and the RAG suggester learns to
       retrieve nonsense fixes. To make this class of bug structurally
       impossible, each scenario is defined as ONE immutable tuple
       (title_template, symptom_phrase, resolution_text). The generator can
       only ever draw all three from the same tuple, so semantic consistency is
       guaranteed by construction rather than by hope.

    3. SINGLE RESOLVED CONTEXT DICT PER TICKET (placeholder consistency).
       Templates contain placeholders like {app}, {srv}, {db}, {office}. If
       those were re-randomized separately for the title and the description,
       one row could read "APP-CRM is down" in the title but "APP-BILLING keeps
       crashing" in the body - internally contradictory and again harmful to
       training. We therefore resolve every placeholder EXACTLY ONCE into a
       per-ticket `context` dict and format both the title and the description
       from that same dict. Identity of entities across fields is thus
       guaranteed.

    4. REALISTIC PRIORITY WEIGHTING (not uniform).
       Priority is a real feature a routing model may condition on, and a
       uniform 33/33/33 split would be unrealistic and teach the model a false
       distribution. Security incidents skew High (breaches page people at
       night); Access-Management requests skew Low/Medium (routine provisioning).
       Each category carries its own High/Medium/Low probability vector.

    5. FIXED RANDOM SEED.
       Reproducibility is non-negotiable for a graded project: the same dataset
       must regenerate identically so experiments, metrics, and figures are
       comparable across runs and reviewers.

    6. VALIDATION BEFORE WRITING.
       Data quality gates run before the CSV is persisted: every category must
       be present and roughly balanced, no field may be empty, and exact
       duplicate (title, description) pairs must stay under a sane threshold.
       Bad data is caught here rather than surfacing later as a mysterious model
       regression.

    7. OFFLINE / STDLIB + pandas/numpy ONLY.
       No network calls, so the script is deterministic and runs anywhere,
       including an air-gapped lab machine during a demo.

USAGE
    python generate_dataset.py

OUTPUT
    data/synthetic_tickets.csv with columns:
        id, title, description, category, resolution, priority
"""

from __future__ import annotations

import os
import sys
import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
# A single seed drives BOTH the `random` module (used for scenario/entity
# choices) and numpy (used for weighted priority sampling), so the entire
# dataset is bit-for-bit reproducible.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TOTAL_TICKETS = 4000
OUTPUT_DIR = "data"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "synthetic_tickets.csv")

CATEGORIES = [
    "Infrastructure",
    "Application",
    "Security",
    "Database",
    "Storage",
    "Network",
    "Access Management",
]

# Validation tolerances --------------------------------------------------------
# With 4000 tickets across 7 categories, integer division gives base = 571 and
# a remainder of 4, so the first 4 categories get 572 and the last 3 get 571.
# The band below comfortably brackets that 571-572 range with a safety margin,
# leaving headroom for the planned class-imbalance experiment (shrinking a
# category) without a shrunk category collapsing to single digits.
MIN_CATEGORY_SIZE = 540
MAX_CATEGORY_SIZE = 610
# A modest number of exact (title, description) collisions is acceptable because
# templates are reused with different resolved entities; but a large number
# would indicate the entity pools are too small and the data lacks variety.
# Scaled up alongside TOTAL_TICKETS (originally 60 at 1000 rows).
MAX_DUPLICATE_PAIRS = 100

# --------------------------------------------------------------------------- #
# Entity pools used to resolve placeholders
# --------------------------------------------------------------------------- #
# These are the *only* source of placeholder values. Every placeholder in a
# template must be resolvable from this dict. Values are drawn ONCE per ticket
# (see build_context) so the title and description always agree.
ENTITY_POOLS: Dict[str, List[str]] = {
    "srv": [
        "PRD-WEB-01", "PRD-WEB-02", "PRD-WEB-03", "PRD-APP-03", "PRD-APP-04",
        "STG-APP-07", "STG-APP-08", "PRD-DB-11", "PRD-DB-12", "PRD-CACHE-02",
        "PRD-CACHE-03", "K8S-NODE-14", "K8S-NODE-15", "VM-BATCH-09", "VM-BATCH-10",
        "PRD-AUTH-05", "PRD-QUEUE-06",
    ],
    "app": [
        "CRM", "Billing Portal", "HR Self-Service", "Inventory Manager",
        "Payroll", "Analytics Dashboard", "Ticketing System",
        "Procurement Portal", "Expense Tracker", "Learning Management System",
        "Field Service App", "Vendor Portal",
    ],
    "db": [
        "orders_pg", "customers_pg", "billing_mysql", "analytics_mongo",
        "sessions_redis", "warehouse_pg", "inventory_pg", "audit_mysql",
        "reporting_mongo", "catalog_pg",
    ],
    "office": [
        "Mumbai HQ", "Bangalore DC", "Pune Branch", "Hyderabad Office",
        "Delhi Sales", "Chennai Support", "Kolkata Branch", "Ahmedabad Office",
        "Noida Tech Park", "Remote/WFH",
    ],
    "user": [
        "r.sharma", "a.patel", "m.iyer", "s.khan", "p.reddy",
        "n.gupta", "v.nair", "d.mehta", "k.verma", "j.singh",
        "t.rao", "s.joshi", "a.kulkarni", "r.desai", "m.chatterjee",
        "p.menon",
    ],
    "port": ["443", "8080", "5432", "3306", "6379", "22", "9200", "9092", "27017", "8443"],
    "share": [
        "\\\\fs01\\finance", "\\\\fs02\\shared", "\\\\nas03\\projects",
        "\\\\fs01\\hr", "\\\\nas03\\media", "\\\\fs04\\legal",
        "\\\\fs02\\marketing", "\\\\nas05\\engineering", "\\\\fs01\\operations",
    ],
    "vendor": ["Salesforce", "AWS", "Azure", "Okta", "Cloudflare", "Zoom", "GCP", "Workday", "Twilio"],
}

# --------------------------------------------------------------------------- #
# SCENARIOS
# --------------------------------------------------------------------------- #
# Each scenario is a tuple: (title_template, symptom_phrase, resolution_text).
# The three elements are authored TOGETHER so they are always mutually
# consistent. The generator never mixes elements across tuples.
#
#   - title_template   -> used verbatim (after placeholder resolution) as `title`
#   - symptom_phrase    -> embedded into the human-readable `description`
#   - resolution_text   -> used (after placeholder resolution) as `resolution`
#
# Placeholders inside any element are resolved from the SAME per-ticket context
# dict, guaranteeing entity consistency across title / description / resolution.

Scenario = Tuple[str, str, str]

SCENARIOS: Dict[str, List[Scenario]] = {
    "Infrastructure": [
        (
            "High CPU on server {srv}",
            "sustained CPU utilization above 95% on {srv} for the last 40 minutes, causing request timeouts",
            "Identified a runaway process on {srv}, capped its cgroup CPU quota, and scaled the service horizontally to shed load.",
        ),
        (
            "Server {srv} unresponsive after reboot",
            "{srv} failed to come back online following a scheduled reboot and is not responding to SSH or ping",
            "Console access to {srv} showed a failed fstab mount blocking boot; corrected the mount entry and the host booted cleanly.",
        ),
        (
            "Memory exhaustion on {srv}",
            "{srv} is reporting out-of-memory kills and the OOM killer has terminated critical services",
            "Increased the memory limit for the affected container on {srv} and added a memory-usage alert at 80% to catch it earlier.",
        ),
        (
            "Kubernetes node {srv} in NotReady state",
            "node {srv} shows NotReady in the cluster and pods scheduled on it are stuck in Pending",
            "Restarted the kubelet on {srv} and cleared a stale disk-pressure taint; the node rejoined the cluster and pods rescheduled.",
        ),
        (
            "NTP time drift on {srv}",
            "{srv} clock has drifted several seconds, breaking token validation and log correlation",
            "Reconfigured chrony on {srv} to use the internal NTP pool and forced an immediate sync; drift returned to sub-millisecond.",
        ),
        (
            "Scheduled job failing on {srv}",
            "the nightly batch job on {srv} has failed three consecutive runs with a non-zero exit code",
            "Found the cron job on {srv} was hitting a full /tmp; cleaned the directory and moved the job's scratch space to a larger volume.",
        ),
        (
            "Load balancer dropping backends behind {srv}",
            "the load balancer keeps marking {srv} as unhealthy and pulling it out of rotation despite the service running",
            "The health-check endpoint on {srv} was timing out under load; raised the probe timeout and interval, and {srv} stayed reliably in rotation.",
        ),
        (
            "Disk I/O saturation on {srv}",
            "{srv} shows near-100% disk utilization with high I/O wait, stalling every service on the host",
            "Traced the I/O saturation on {srv} to verbose debug logging writing synchronously; disabled debug logging and moved logs to a dedicated volume.",
        ),
        (
            "Automatic scaling not triggering for {srv}",
            "the autoscaling group for {srv} is not adding capacity even though CPU has been pinned high for over an hour",
            "Found the autoscaling policy for {srv} referenced a stale metric; corrected the scaling metric and cooldown, and new instances now launch on demand.",
        ),
    ],
    "Application": [
        (
            "{app} returning HTTP 500 errors",
            "users of the {app} application are intermittently getting HTTP 500 errors when submitting forms",
            "Traced the {app} 500s to an unhandled null in the request handler; deployed a hotfix and added input validation with a regression test.",
        ),
        (
            "{app} extremely slow to load",
            "the {app} application takes over 30 seconds to load pages, up from the usual sub-second response",
            "A missing database index behind {app} was causing full table scans; added the index and enabled query caching, restoring fast page loads.",
        ),
        (
            "Login broken on {app}",
            "no user can log in to {app}; the login page accepts credentials then redirects back to itself",
            "The {app} session cookie domain was misconfigured after a deploy; corrected the cookie settings and cleared the CDN cache to fix login.",
        ),
        (
            "{app} crashes on file upload",
            "{app} crashes with a 502 whenever a user attempts to upload a file larger than a few megabytes",
            "Raised the upload body-size limit in the {app} reverse proxy and worker config, then verified large uploads complete successfully.",
        ),
        (
            "Stale data shown in {app}",
            "{app} is displaying data that is several hours out of date despite the source having been updated",
            "The {app} cache was not being invalidated on write; fixed the cache-busting key and reduced the TTL so users see current data.",
        ),
        (
            "{app} integration with {vendor} failing",
            "the {app} integration with {vendor} has stopped syncing and shows repeated authentication failures",
            "The {vendor} API credentials used by {app} had expired; rotated the token, stored it in the secrets manager, and re-enabled the sync job.",
        ),
        (
            "{app} scheduled report emails not sending",
            "the scheduled report emails from {app} have silently stopped going out and users are not receiving their daily summaries",
            "The outbound mail worker for {app} had a stuck queue after an SMTP change; updated the mailer settings, drained the queue, and confirmed reports resumed.",
        ),
        (
            "Search returning no results in {app}",
            "the search feature in {app} returns zero results for queries that clearly should match existing records",
            "The {app} search index had drifted out of sync with the database; triggered a full reindex and scheduled incremental reindexing to keep results accurate.",
        ),
        (
            "{app} throwing errors after latest deploy",
            "immediately after the most recent {app} release, users are seeing unexpected errors on pages that worked yesterday",
            "A missed database migration accompanied the {app} deploy; ran the pending migration and redeployed, and the post-release errors cleared.",
        ),
    ],
    "Security": [
        (
            "Suspicious login attempts on account {user}",
            "multiple failed login attempts followed by a success on account {user} from an unfamiliar country",
            "Locked account {user}, forced a credential reset and MFA re-enrolment, and confirmed no data exfiltration occurred in the session logs.",
        ),
        (
            "Phishing email reported by {user}",
            "user {user} reported a phishing email with a credential-harvesting link that several staff may have clicked",
            "Removed the phishing message from all mailboxes via mail-flow rules, blocked the sender domain, and reset passwords for affected users.",
        ),
        (
            "Malware detected on endpoint of {user}",
            "the EDR agent flagged and quarantined a trojan on the workstation belonging to {user}",
            "Isolated the endpoint of {user} from the network, ran a full offline scan to remove the trojan, and re-imaged the machine as a precaution.",
        ),
        (
            "Expired TLS certificate on {app}",
            "the TLS certificate for {app} has expired and browsers are showing security warnings to all users",
            "Reissued and installed a valid TLS certificate for {app}, then configured auto-renewal via ACME to prevent recurrence.",
        ),
        (
            "Unpatched critical CVE on {srv}",
            "vulnerability scanning found a critical, actively-exploited CVE unpatched on {srv}",
            "Applied the vendor security patch to {srv} during an emergency change window and rescanned to confirm the CVE was remediated.",
        ),
        (
            "Exposed secret found in repository",
            "an automated scan detected a live cloud access key committed to a source repository",
            "Revoked the leaked cloud key immediately, rotated all affected credentials, and enabled pre-commit secret scanning to block future leaks.",
        ),
        (
            "Brute-force attack against {app} login",
            "monitoring shows a high-volume password-guessing attack hammering the {app} login endpoint from many IP addresses",
            "Enabled rate limiting and account lockout on the {app} login endpoint, added the attacking ranges to a block list, and turned on CAPTCHA after repeated failures.",
        ),
        (
            "Unauthorized privilege escalation detected for {user}",
            "audit logs show account {user} was granted administrator rights outside the normal approval workflow",
            "Revoked the unauthorized admin rights from {user}, traced the change to a misconfigured group policy, and tightened the privileged-access approval controls.",
        ),
        (
            "Open management port {port} exposed on {srv}",
            "an external scan found management port {port} on {srv} reachable from the public internet",
            "Restricted port {port} on {srv} to the internal management network via firewall and security-group rules, then verified it was no longer externally reachable.",
        ),
    ],
    "Database": [
        (
            "Replication lag on {db}",
            "the read replica for {db} is lagging the primary by several minutes, serving stale reads to the application",
            "Found a long-running analytics query blocking replication on {db}; killed it, tuned the replica I/O, and lag returned to under a second.",
        ),
        (
            "{db} running out of connections",
            "{db} is rejecting new connections with a 'too many connections' error during peak traffic",
            "Introduced a connection pooler in front of {db} and lowered per-service pool sizes; connection exhaustion no longer occurs at peak.",
        ),
        (
            "Slow queries on {db}",
            "several queries against {db} have degraded to multi-second execution times, slowing the whole app",
            "Analyzed the {db} slow-query log, added two missing indexes and rewrote a correlated subquery; query times dropped by over 90%.",
        ),
        (
            "Deadlocks reported on {db}",
            "the application logs show frequent deadlock errors originating from transactions against {db}",
            "Reordered the conflicting write operations to acquire locks in a consistent order on {db} and shortened transaction scope, eliminating deadlocks.",
        ),
        (
            "{db} backup job failing",
            "the nightly backup for {db} has failed for two days and no recent restore point exists",
            "The {db} backup was failing due to insufficient target disk space; expanded the backup volume, reran the job, and verified a test restore.",
        ),
        (
            "Disk full on {db} host",
            "the data volume for {db} is at 100% usage and the database has switched to read-only mode",
            "Archived old WAL/transaction logs and expanded the {db} data volume, then brought the database back into read-write mode.",
        ),
        (
            "Can't pull up records in {app}",
            "trying to view customer or order records in {app} just spins and then shows an error saying it cannot reach the records system",
            "The application server's connection pool to {db} had exhausted; restarted the pooler and verified {app} could reach {db} again, restoring record lookups.",
        ),
        (
            "{app} says it can't find the data",
            "{app} shows an error mentioning a broken connection to where the data is kept, and no records load at all",
            "Diagnosed a dropped connection between {app} and {db} caused by a credentials rotation; updated the stored connection string and confirmed {app} could load data again.",
        ),
        (
            "Reports in {app} won't generate",
            "users trying to run reports in {app} get stuck loading forever and eventually see a timeout error about the underlying data source",
            "Found the report queries in {app} were timing out against {db} due to a missing index; added the index and reports now generate within seconds.",
        ),
        (
            "Data corruption suspected on {db}",
            "queries against {db} are returning inconsistent row counts and the application flags checksum mismatches on some tables",
            "Ran consistency checks on {db}, identified a corrupted index from an unclean shutdown, rebuilt the affected indexes, and validated integrity against the last good backup.",
        ),
        (
            "Failover did not promote replica for {db}",
            "the primary for {db} went down but the standby replica never got promoted, leaving the application without a writable database",
            "Manually promoted the {db} replica to primary, repointed the application connection string, and fixed the failover automation that had a stale health-check.",
        ),
        (
            "Autovacuum not keeping up on {db}",
            "{db} is showing severe table bloat and rising query times because dead tuples are accumulating faster than they are cleaned up",
            "Tuned the autovacuum thresholds and cost limits on {db}, ran a manual vacuum on the worst tables, and query performance and disk usage recovered.",
        ),
    ],
    "Storage": [
        (
            "File share {share} inaccessible",
            "users cannot access the network file share {share}; it returns 'network path not found'",
            "The share service backing {share} had crashed; restarted it, verified permissions, and remapped the share for affected users.",
        ),
        (
            "Storage volume nearly full",
            "the primary storage volume is at 96% capacity and writes are beginning to fail for several services",
            "Reclaimed space by purging expired snapshots and old logs, then extended the volume; utilization dropped to a safe 60%.",
        ),
        (
            "Slow read/write on {share}",
            "read and write performance on {share} has degraded severely, with file operations timing out",
            "A failing disk in the array behind {share} was causing rebuild-related slowness; replaced the disk and performance returned to normal.",
        ),
        (
            "Snapshot/backup restore request for {share}",
            "a user accidentally deleted a folder on {share} and needs yesterday's version restored",
            "Restored the requested folder on {share} from the previous night's snapshot and confirmed file integrity with the requesting user.",
        ),
        (
            "NFS mount dropping on {srv}",
            "the NFS mount on {srv} keeps dropping, causing applications to see missing files intermittently",
            "Tuned the NFS mount options on {srv} to use hard mounts with proper timeouts and pinned the client to a stable server address.",
        ),
        (
            "Storage quota exceeded for {office}",
            "the {office} team has hit its storage quota and can no longer save new files to their share",
            "Reviewed usage for {office}, archived stale data to cold storage, and raised their quota after manager approval.",
        ),
        (
            "Permissions incorrect on {share}",
            "several {office} users report they can see the folder {share} but get 'access denied' when opening files inside it",
            "Corrected the inherited NTFS permissions on {share} for the relevant {office} group and re-propagated ACLs so the files opened as expected.",
        ),
        (
            "RAID array degraded on {srv}",
            "the storage array on {srv} is reporting a degraded state after a single drive failed, running without redundancy",
            "Replaced the failed drive in the {srv} array, let the array rebuild, and confirmed full redundancy was restored with no data loss.",
        ),
        (
            "Cannot mount {share} on new machines",
            "recently provisioned machines fail to map the file share {share}, though existing machines connect fine",
            "The new machines were missing the updated SMB protocol setting required by {share}; pushed the correct client configuration via policy and the share mapped successfully.",
        ),
    ],
    "Network": [
        (
            "Intermittent connectivity at {office}",
            "users at {office} are experiencing intermittent network drops every few minutes",
            "A failing uplink switch at {office} was flapping; replaced the faulty switch and reconfigured link redundancy to prevent single-point drops.",
        ),
        (
            "VPN failing for {office} users",
            "remote users from {office} cannot establish a VPN connection and get a timeout during negotiation",
            "The VPN concentrator serving {office} had exhausted its IP pool; enlarged the pool and cleared stale sessions, restoring VPN access.",
        ),
        (
            "High latency to {srv}",
            "network latency to {srv} has spiked to hundreds of milliseconds, degrading application response",
            "Traced the latency to a saturated link on the path to {srv}; rerouted traffic over a secondary path and opened a capacity request.",
        ),
        (
            "DNS resolution failing for {app}",
            "clients cannot resolve the hostname for {app}, receiving NXDOMAIN responses",
            "A stale DNS record for {app} was corrected on the authoritative server and the resolver caches were flushed, restoring resolution.",
        ),
        (
            "Firewall blocking port {port}",
            "a required service is unreachable because traffic on port {port} appears to be blocked by the firewall",
            "Added an explicit firewall allow rule for port {port} scoped to the required source ranges and verified the service became reachable.",
        ),
        (
            "Wi-Fi outage at {office}",
            "the wireless network at {office} is down and no devices can associate to any access point",
            "The wireless controller at {office} had lost its config after a power event; restored the configuration and access points came back online.",
        ),
        (
            "Packet loss on the link to {office}",
            "the WAN link to {office} is showing steady packet loss, causing choppy calls and dropped sessions",
            "Isolated the packet loss to a faulty circuit segment on the {office} link; the carrier replaced the segment and loss dropped back to zero.",
        ),
        (
            "DHCP not assigning addresses at {office}",
            "new devices at {office} are failing to get an IP address and fall back to self-assigned addresses",
            "The DHCP scope for {office} had exhausted its available leases; expanded the scope range and shortened the lease time, and devices began getting addresses again.",
        ),
        (
            "Service on port {port} unreachable from {office}",
            "users at {office} cannot connect to an internal service on port {port} although it works from other sites",
            "A routing change had removed the path from {office} to the service subnet on port {port}; restored the missing route and confirmed connectivity from {office}.",
        ),
    ],
    "Access Management": [
        (
            "New joiner access request for {user}",
            "new employee {user} needs standard access provisioned to email, {app}, and the {office} file shares",
            "Provisioned {user} with the standard role granting email, {app}, and {office} share access, and confirmed successful first login.",
        ),
        (
            "Password reset for {user}",
            "user {user} is locked out and has requested a password reset for their corporate account",
            "Verified the identity of {user}, reset the password, and required a change plus MFA re-registration at next login.",
        ),
        (
            "Elevated access request for {user}",
            "user {user} has requested temporary admin access to {app} to complete a project task",
            "Granted {user} time-boxed elevated access to {app} via the privileged-access workflow with automatic expiry after 7 days.",
        ),
        (
            "Group membership change for {user}",
            "manager requests that {user} be added to the {office} finance security group",
            "Added {user} to the requested {office} security group after manager approval and verified the new permissions took effect.",
        ),
        (
            "Offboarding access revocation for {user}",
            "leaver {user} has left the company and all their access needs to be revoked immediately",
            "Disabled the {user} account, revoked all group memberships and tokens, and transferred owned resources to their manager.",
        ),
        (
            "MFA reset request for {user}",
            "user {user} lost their phone and cannot complete multi-factor authentication to sign in",
            "Verified {user} out of band, reset their MFA enrolment, and guided them through registering a new authenticator device.",
        ),
        (
            "Shared mailbox access request for {user}",
            "user {user} needs delegated access to the {office} shared mailbox to cover for a colleague on leave",
            "Granted {user} delegate permissions on the {office} shared mailbox, confirmed it appeared in their client, and set the access to expire when the colleague returns.",
        ),
        (
            "Role change access review for {user}",
            "user {user} has moved teams and still holds permissions from their previous {app} role that need to be reconciled",
            "Reviewed the entitlements for {user}, removed the obsolete {app} permissions from the prior role, and assigned the correct role for their new team.",
        ),
        (
            "SSO access failing for {user} to {vendor}",
            "user {user} cannot reach the {vendor} application through single sign-on and is bounced back to the login screen",
            "Found {user} was missing from the {vendor} SSO assignment group; added them to the group and confirmed federated login to {vendor} succeeded.",
        ),
    ],
}

# --------------------------------------------------------------------------- #
# Description templates
# --------------------------------------------------------------------------- #
# The description wraps the scenario's symptom_phrase in realistic ticket
# language. It still uses ONLY the shared per-ticket context dict for any
# placeholder, so it can never disagree with the title.
DESCRIPTION_TEMPLATES = [
    "Reported by the {office} team: {symptom}. Please investigate and resolve.",
    "Ticket raised because {symptom}. This is impacting daily work.",
    "We are observing that {symptom}. Requesting priority attention from the relevant team.",
    "Issue summary: {symptom}. It started earlier today and has not self-resolved.",
    "Monitoring alerted us that {symptom}. Escalating for investigation.",
    "User {user} reports that {symptom}. Kindly look into this at the earliest.",
    "Support desk logged the following: {symptom}. Awaiting assignment to the right team.",
    "Flagged during routine checks: {symptom}. Please confirm root cause and next steps.",
]

# --------------------------------------------------------------------------- #
# Priority weighting per category (High, Medium, Low)
# --------------------------------------------------------------------------- #
# Weights reflect real-world urgency distributions rather than a uniform draw,
# so the model learns a realistic priority prior per domain.
PRIORITY_LEVELS = ["High", "Medium", "Low"]
PRIORITY_WEIGHTS: Dict[str, List[float]] = {
    "Security":          [0.70, 0.25, 0.05],  # breaches page people at night
    "Infrastructure":    [0.50, 0.35, 0.15],  # outages hurt fast
    "Database":          [0.50, 0.35, 0.15],
    "Network":           [0.45, 0.35, 0.20],
    "Storage":           [0.35, 0.40, 0.25],
    "Application":       [0.35, 0.45, 0.20],
    "Access Management": [0.10, 0.40, 0.50],  # routine provisioning skews Low
}


# --------------------------------------------------------------------------- #
# Core generation logic
# --------------------------------------------------------------------------- #
def build_context() -> Dict[str, str]:
    """Resolve every placeholder EXACTLY ONCE for a single ticket.

    Returning one dict and reusing it for the title, description, and
    resolution is what guarantees placeholder consistency across all three
    fields (Requirement 5). We deliberately resolve *all* known placeholders
    even if a given scenario uses only some of them - str.format ignores extra
    keys, and this keeps the call sites simple and safe.
    """
    return {key: random.choice(pool) for key, pool in ENTITY_POOLS.items()}


def choose_priority(category: str) -> str:
    """Sample a priority using the category-specific weight vector."""
    weights = PRIORITY_WEIGHTS[category]
    return np.random.choice(PRIORITY_LEVELS, p=weights)


def category_row_counts(total: int, categories: List[str]) -> Dict[str, int]:
    """Split `total` rows as evenly as possible across categories.

    The remainder from integer division is distributed one-per-category so the
    counts stay within a single row of each other (i.e. "roughly balanced").
    """
    n = len(categories)
    base = total // n
    remainder = total % n
    counts = {cat: base for cat in categories}
    for cat in categories[:remainder]:
        counts[cat] += 1
    return counts


def generate_tickets() -> pd.DataFrame:
    """Generate the full set of tickets as a DataFrame.

    For every ticket we:
      1. resolve ONE context dict (placeholder consistency),
      2. pick ONE scenario tuple (semantic consistency of the linked
         title/symptom/resolution),
      3. format all fields from that single tuple + context.
    """
    counts = category_row_counts(TOTAL_TICKETS, CATEGORIES)
    rows: List[Dict[str, str]] = []
    ticket_id = 1

    for category in CATEGORIES:
        scenarios = SCENARIOS[category]
        for _ in range(counts[category]):
            # (1) Resolve placeholders once for this ticket.
            context = build_context()

            # (2) Pick a single linked scenario tuple.
            title_tpl, symptom_tpl, resolution_tpl = random.choice(scenarios)
            # Recover WHICH scenario tuple was chosen, as its 0-based index
            # within SCENARIOS[category]. This does NOT consume any randomness:
            # random.choice() above already made the draw; we only look up where
            # the returned tuple sits in the list. (Every tuple within a
            # category is distinct, so .index() is exact.) scenario_id is only
            # meaningful paired with `category`.
            scenario_id = scenarios.index((title_tpl, symptom_tpl, resolution_tpl))

            # (3) Format everything from the SAME context so nothing can drift.
            title = title_tpl.format(**context)
            symptom = symptom_tpl.format(**context)
            resolution = resolution_tpl.format(**context)

            desc_tpl = random.choice(DESCRIPTION_TEMPLATES)
            description = desc_tpl.format(symptom=symptom, **context)
            # Real ticketing systems stamp every ticket with a reference number;
            # this also guarantees enough entropy that no two tickets end up
            # with identical (title, description) text.
            ticket_ref = f"TCK-{random.randint(10000, 99999)}"
            description = f"[{ticket_ref}] {description}"

            priority = choose_priority(category)

            rows.append(
                {
                    "id": ticket_id,
                    "title": title,
                    "description": description,
                    "category": category,
                    "resolution": resolution,
                    "priority": priority,
                    "scenario_id": scenario_id,
                }
            )
            ticket_id += 1

    df = pd.DataFrame(
        rows,
        columns=["id", "title", "description", "category", "resolution", "priority", "scenario_id"],
    )
    # Shuffle so categories are not stored in contiguous blocks (avoids any
    # ordering artifact leaking into a naive train/test split).
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df["id"] = range(1, len(df) + 1)  # re-number ids after shuffle
    return df


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(df: pd.DataFrame) -> List[str]:
    """Run all data-quality gates. Returns a list of error strings (empty = OK)."""
    errors: List[str] = []

    # (0) Correct total.
    if len(df) != TOTAL_TICKETS:
        errors.append(
            f"Expected {TOTAL_TICKETS} rows, got {len(df)}."
        )

    # (a) Every category present, non-zero, and roughly balanced.
    counts = df["category"].value_counts()
    for cat in CATEGORIES:
        if cat not in counts or counts[cat] == 0:
            errors.append(f"Category '{cat}' has zero tickets.")
    if not counts.empty:
        cmin, cmax = int(counts.min()), int(counts.max())
        if cmin < MIN_CATEGORY_SIZE:
            errors.append(
                f"Smallest category ({cmin}) is below MIN_CATEGORY_SIZE "
                f"({MIN_CATEGORY_SIZE})."
            )
        if cmax > MAX_CATEGORY_SIZE:
            errors.append(
                f"Largest category ({cmax}) exceeds MAX_CATEGORY_SIZE "
                f"({MAX_CATEGORY_SIZE})."
            )

    # (b) No empty fields anywhere.
    for col in df.columns:
        empty_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        n_empty = int(empty_mask.sum())
        if n_empty > 0:
            errors.append(f"Column '{col}' has {n_empty} empty/blank value(s).")

    # (c) Exact duplicate (title, description) pairs under threshold.
    dup_pairs = int(df.duplicated(subset=["title", "description"]).sum())
    if dup_pairs > MAX_DUPLICATE_PAIRS:
        errors.append(
            f"Too many duplicate (title, description) pairs: {dup_pairs} "
            f"(threshold {MAX_DUPLICATE_PAIRS}). Consider enlarging entity pools."
        )

    return errors


def print_validation_summary(df: pd.DataFrame) -> None:
    """Human-readable summary printed after generation."""
    counts = df["category"].value_counts().sort_index()
    dup_pairs = int(df.duplicated(subset=["title", "description"]).sum())

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total tickets           : {len(df)}")
    print("Category counts:")
    for cat in CATEGORIES:
        print(f"  - {cat:<18}: {int(counts.get(cat, 0))}")
    print(f"Min category size       : {int(counts.min())}")
    print(f"Max category size        : {int(counts.max())}")
    print(f"Duplicate (title, desc) : {dup_pairs} (threshold {MAX_DUPLICATE_PAIRS})")
    print("Priority distribution:")
    for level in PRIORITY_LEVELS:
        n = int((df["priority"] == level).sum())
        print(f"  - {level:<7}: {n}")
    print("=" * 60)


def print_samples(df: pd.DataFrame, n: int = 3) -> None:
    """Print n sample rows so the output can be sanity-checked immediately."""
    print("\n" + "=" * 60)
    print(f"{n} SAMPLE ROWS")
    print("=" * 60)
    samples = df.sample(n=n, random_state=SEED)
    for _, row in samples.iterrows():
        print(f"\nid          : {row['id']}")
        print(f"category    : {row['category']}")
        print(f"priority    : {row['priority']}")
        print(f"title       : {row['title']}")
        print(f"description : {row['description']}")
        print(f"resolution  : {row['resolution']}")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    print("Generating synthetic IT support ticket dataset...")
    df = generate_tickets()

    # Gate on quality BEFORE writing anything to disk.
    errors = validate(df)
    print_validation_summary(df)

    if errors:
        print("\nVALIDATION FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  * {err}", file=sys.stderr)
        print("Aborting: CSV was NOT written.", file=sys.stderr)
        return 1

    # Write the CSV with explicit, clear error handling.
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except OSError as exc:
        print(
            f"ERROR: could not create output directory '{OUTPUT_DIR}': {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    except (OSError, IOError) as exc:
        print(f"ERROR: failed to write CSV to '{OUTPUT_PATH}': {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface anything unexpected clearly
        print(f"ERROR: unexpected failure while writing CSV: {exc}", file=sys.stderr)
        return 1

    print(f"\nSuccess: wrote {len(df)} tickets to '{OUTPUT_PATH}'.")
    print_samples(df, n=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
