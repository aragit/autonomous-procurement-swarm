#!/usr/bin/env bash
# ==============================================================================
# Autonomous Procurement Swarm — Comprehensive Terminal Test Suite
# ==============================================================================
# Prerequisites:
#   docker compose up -d   (Postgres on 5433 + API on 8000)
# ==============================================================================

set -euo pipefail

BASE_URL="http://localhost:8000"
PASS=0
FAIL=0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_pass() { echo -e "${GREEN}✅ PASS${NC}: $1"; PASS=$((PASS+1)); }
log_fail() { echo -e "${RED}❌ FAIL${NC}: $1"; FAIL=$((FAIL+1)); }
log_info() { echo -e "${YELLOW}ℹ️  INFO${NC}: $1"; }

http_post() { curl -s -X POST "$BASE_URL$1" -H "Content-Type: application/json" -d "$2"; }
http_get()  { curl -s -X GET "$BASE_URL$1"; }

assert_json_field() {
    local json="$1" field="$2" expected="$3"
    local actual=$(echo "$json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('$field', '__MISSING__'))" 2>/dev/null || echo "__PARSE_ERROR__")
    if [ "$actual" = "$expected" ]; then log_pass "$4"; else log_fail "$4 (expected: $expected, got: $actual)"; fi
}

assert_json_contains() {
    local json="$1" field="$2"
    local actual=$(echo "$json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('$field','__MISSING__'))" 2>/dev/null || echo "__PARSE_ERROR__")
    if [ "$actual" != "__MISSING__" ] && [ "$actual" != "__PARSE_ERROR__" ]; then
        log_pass "$3"
    else
        log_fail "$3 (field '$field' missing/unparseable)"
    fi
}

assert_http_status() {
    local url="$1" method="${2:-GET}" body="${3:-}" expected="${4:-200}"
    if [ "$method" = "POST" ]; then
        status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL$url" -H "Content-Type: application/json" -d "$body")
    else
        status=$(curl -s -o /dev/null -w "%{http_code}" -X GET "$BASE_URL$url")
    fi
    if [ "$status" = "$expected" ]; then log_pass "$5 (HTTP $status)"; else log_fail "$5 (expected HTTP $expected, got $status)"; fi
}

# ─── SECTION 1: FUNCTIONAL ───
echo ""; echo "================================================================================"; echo "  SECTION 1: FUNCTIONAL TESTS"; echo "================================================================================"; echo ""

log_info "1.1 Health Check"
HEALTH=$(http_get "/health")
assert_json_field "$HEALTH" "status" "healthy" "Health returns healthy"
assert_json_field "$HEALTH" "database" "connected" "Database connected"

log_info "1.2 Basic Auction (steel, bartering)"
AUCTION1=$(http_post "/auctions" '{"material":"steel","quantity":1000,"supplier_count":5,"enable_bartering":true}')
assert_json_contains "$AUCTION1" "session_id" "Auction returns session_id"
assert_json_field "$AUCTION1" "status" "AWARDED" "Auction AWARDED"
SESSION1=$(echo "$AUCTION1" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
log_info "Session 1: $SESSION1"

log_info "1.3 Ledger Chain Integrity"
LEDGER1=$(http_get "/auctions/$SESSION1")
assert_json_field "$LEDGER1" "chain_valid" "True" "Chain valid"
assert_json_contains "$LEDGER1" "events" "Ledger has events"

log_info "1.4 Auction Stats"
STATS1=$(http_get "/auctions/$SESSION1/stats")
assert_json_contains "$STATS1" "total_events" "Stats total_events"
assert_json_contains "$STATS1" "bids_received" "Stats bids_received"

log_info "1.5 Auction Without Bartering"
AUCTION2=$(http_post "/auctions" '{"material":"aluminum","quantity":500,"supplier_count":3,"enable_bartering":false}')
assert_json_field "$AUCTION2" "status" "AWARDED" "Non-bartering AWARDED"
assert_json_contains "$AUCTION2" "winner" "Has winner"

log_info "1.6 Different Materials"
for mat in copper plastic lumber rubber; do
    RESP=$(http_post "/auctions" "{\"material\":\"$mat\",\"quantity\":100,\"supplier_count\":3,\"enable_bartering\":false}")
    assert_json_field "$RESP" "status" "AWARDED" "Material $mat AWARDED"
done

log_info "1.7 Global Ledger Stats"
GLOBAL=$(http_get "/ledger/stats")
assert_json_contains "$GLOBAL" "total_events" "Global total_events"
assert_json_contains "$GLOBAL" "total_sessions" "Global total_sessions"
assert_json_contains "$GLOBAL" "deals_awarded" "Global deals_awarded"

log_info "1.8 Supplier Profiles"
SUPPLIERS=$(http_get "/suppliers")
assert_json_contains "$SUPPLIERS" "suppliers" "Suppliers list"
# Pick a supplier that actually has a profile (first in list)
FIRST_SUPPLIER=$(echo "$SUPPLIERS" | python3 -c "import sys,json; s=json.load(sys.stdin).get('suppliers',[]); print(s[0].get('supplier_id','') if s else '')" 2>/dev/null)

log_info "1.9 Individual Supplier Profile"
if [ -n "$FIRST_SUPPLIER" ]; then
    PROFILE=$(http_get "/suppliers/$FIRST_SUPPLIER/profile")
    assert_json_contains "$PROFILE" "supplier_id" "Profile has supplier_id"
    assert_json_contains "$PROFILE" "concession_speed" "Profile has concession_speed"
else
    log_fail "No suppliers with profiles found for 1.9"
fi

log_info "1.10 Supplier Similarity Search"
if [ -n "$FIRST_SUPPLIER" ]; then
    SIMILAR=$(http_get "/suppliers/similar?supplier_id=$FIRST_SUPPLIER&n=3")
    assert_json_contains "$SIMILAR" "similar" "Similarity search returns similar"
else
    log_fail "No supplier available for similarity test"
fi


log_info "1.11 Invalid Material"
assert_http_status "/auctions" "POST" '{"material":"unobtainium","quantity":100,"supplier_count":3}' "422" "Invalid material 422"

log_info "1.12 Zero Quantity"
assert_http_status "/auctions" "POST" '{"material":"steel","quantity":0,"supplier_count":3}' "422" "Zero qty 422"

log_info "1.13 Non-existent Session"
assert_http_status "/auctions/nonexistent123" "GET" "" "404" "Missing session 404"

log_info "1.14 Negative Supplier Count"
assert_http_status "/auctions" "POST" '{"material":"steel","quantity":100,"supplier_count":-1}' "422" "Negative count 422"

# ─── SECTION 2: RESILIENCE ───
echo ""; echo "================================================================================"; echo "  SECTION 2: RESILIENCE TESTS"; echo "================================================================================"; echo ""

log_info "2.1 10 Concurrent Auctions"
START_TIME=$(date +%s)
for i in {1..10}; do http_post "/auctions" '{"material":"steel","quantity":100,"supplier_count":3,"enable_bartering":false}' > /tmp/auction_$i.json & done
wait
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
log_pass "10 concurrent completed in ${DURATION}s"
SUCCESS_COUNT=0
for i in {1..10}; do
    STATUS=$(python3 -c "import json; print(json.load(open('/tmp/auction_$i.json')).get('status','ERROR'))" 2>/dev/null || echo "ERROR")
    [ "$STATUS" = "AWARDED" ] && ((SUCCESS_COUNT++)) || true
done
if [ "$SUCCESS_COUNT" -eq 10 ]; then log_pass "All 10 concurrent AWARDED"; else log_fail "Only $SUCCESS_COUNT/10 succeeded"; fi

log_info "2.2 20 Bartering Auctions (sequential)"
BARTER_SUCCESS=0
for i in {1..20}; do
    RESP=$(http_post "/auctions" '{"material":"copper","quantity":200,"supplier_count":5,"enable_bartering":true}')
    STATUS=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','ERROR'))" 2>/dev/null || echo "ERROR")
    [ "$STATUS" = "AWARDED" ] || [ "$STATUS" = "TERMINATED" ] && ((BARTER_SUCCESS++)) || true
    sleep 0.1
done
if [ "$BARTER_SUCCESS" -eq 20 ]; then log_pass "20 bartering auctions OK"; else log_fail "Only $BARTER_SUCCESS/20"; fi

log_info "2.3 DB Persistence"
BEFORE_SESSIONS=$(echo "$(http_get "/ledger/stats")" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_sessions',0))")
for i in {1..5}; do http_post "/auctions" '{"material":"rubber","quantity":50,"supplier_count":3,"enable_bartering":false}' > /dev/null; done
AFTER_SESSIONS=$(echo "$(http_get "/ledger/stats")" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_sessions',0))")
if [ "$AFTER_SESSIONS" -gt "$BEFORE_SESSIONS" ]; then log_pass "Persisted ($BEFORE_SESSIONS → $AFTER_SESSIONS)"; else log_fail "No persistence"; fi

log_info "2.4 Chain Integrity Under Load"
LATEST_SESSION=$(http_post "/auctions" '{"material":"steel","quantity":100,"supplier_count":3}' | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
CHAIN_VALID=$(echo "$(http_get "/auctions/$LATEST_SESSION")" | python3 -c "import sys,json; print(json.load(sys.stdin).get('chain_valid',False))")
if [ "$CHAIN_VALID" = "True" ]; then log_pass "Chain valid under load"; else log_fail "Chain corrupted"; fi

log_info "2.5 Large Quantity (100,000 units)"
LARGE=$(http_post "/auctions" '{"material":"steel","quantity":100000,"supplier_count":5,"enable_bartering":false}')
LARGE_STATUS=$(echo "$LARGE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','ERROR'))" 2>/dev/null || echo "ERROR")
if [ "$LARGE_STATUS" = "AWARDED" ] || [ "$LARGE_STATUS" = "TERMINATED" ]; then log_pass "Large qty handled"; else log_fail "Large qty failed: $LARGE_STATUS"; fi

log_info "2.6 Memory Accumulation"
for i in {1..3}; do http_post "/auctions" '{"material":"aluminum","quantity":300,"supplier_count":5,"enable_bartering":true}' > /dev/null; done
MEM_COUNT=$(echo "$(http_get "/suppliers")" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('suppliers',[])))" 2>/dev/null || echo "0")
if [ "$MEM_COUNT" -ge 4 ]; then log_pass "Memory accumulated ($MEM_COUNT suppliers)"; else log_fail "Only $MEM_COUNT suppliers"; fi

log_info "2.7 Similarity Consistency"
CONSISTENCY_TARGET="${FIRST_SUPPLIER:-MinerCorp_A}"
SIM1_TOP=$(echo "$(http_get "/suppliers/similar?supplier_id=$CONSISTENCY_TARGET&n=2")" | python3 -c "import sys,json; m=json.load(sys.stdin).get('similar',[]); print(m[0]['supplier_id'] if m else 'NONE')")
SIM2_TOP=$(echo "$(http_get "/suppliers/similar?supplier_id=$CONSISTENCY_TARGET&n=2")" | python3 -c "import sys,json; m=json.load(sys.stdin).get('similar',[]); print(m[0]['supplier_id'] if m else 'NONE')")
if [ "$SIM1_TOP" = "$SIM2_TOP" ]; then log_pass "Similarity deterministic ($SIM1_TOP)"; else log_fail "Inconsistent ($SIM1_TOP vs $SIM2_TOP)"; fi

# ─── SUMMARY ───
echo ""; echo "================================================================================"; echo "  TEST SUMMARY"; echo "================================================================================";
echo -e "  ${GREEN}PASSED${NC}: $PASS"; echo -e "  ${RED}FAILED${NC}: $FAIL"; echo "================================================================================"
if [ "$FAIL" -eq 0 ]; then echo -e "${GREEN}🎉 ALL TESTS PASSED${NC}"; exit 0; else echo -e "${RED}⚠️  SOME TESTS FAILED${NC}"; exit 1; fi
