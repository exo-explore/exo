#!/data/data/com.termux/files/usr/bin/bash
#
# ARM Device Detection & Optimization Script for exo
# ===================================================
# Detects ARM CPU cores, features, and recommends optimal compiler flags.
#
# Based on research from:
#   - docs/ARM_CORTEX_OPTIMIZATION_GUIDE.md
#   - docs/ARM_CLUSTERING_RESEARCH.md
#
# Usage:
#   chmod +x scripts/arm_detect.sh
#   ./scripts/arm_detect.sh              # Full detection
#   ./scripts/arm_detect.sh --flags      # Just output compiler flags
#   ./scripts/arm_detect.sh --json       # Output as JSON
#   source scripts/arm_detect.sh --env   # Export as environment variables
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Output mode
MODE="${1:-full}"

# ============================================================================
# CPU Part Number to Core Name Mapping
# Reference: ARM Cortex documentation
# ============================================================================

declare -A CPU_PART_TO_NAME=(
    # Cortex-X Series (Prime)
    ["0xd85"]="Cortex-X925"
    ["0xd44"]="Cortex-X4"
    ["0xd4e"]="Cortex-X3"
    ["0xd40"]="Cortex-X2"
    ["0xd4c"]="Cortex-X1C"
    ["0xd4b"]="Cortex-X1"
    
    # Cortex-A7xx Series (Big)
    ["0xd47"]="Cortex-A725"
    ["0xd88"]="Cortex-A720"
    ["0xd43"]="Cortex-A715"
    ["0xd41"]="Cortex-A710"
    ["0xd41"]="Cortex-A78"
    ["0xd4d"]="Cortex-A78C"
    ["0xd0d"]="Cortex-A77"
    ["0xd0b"]="Cortex-A76"
    ["0xd0a"]="Cortex-A75"
    ["0xd09"]="Cortex-A73"
    ["0xd08"]="Cortex-A72"
    ["0xd07"]="Cortex-A57"
    
    # Cortex-A5xx Series (Little)
    ["0xd46"]="Cortex-A520"
    ["0xd42"]="Cortex-A510"
    ["0xd05"]="Cortex-A55"
    ["0xd03"]="Cortex-A53"
    ["0xd04"]="Cortex-A35"
    
    # Qualcomm Custom
    ["0x802"]="Kryo-Gold"
    ["0x803"]="Kryo-Silver"
    ["0x804"]="Kryo-485-Gold"
    ["0x805"]="Kryo-485-Silver"
)

declare -A CPU_PART_TO_MCPU=(
    ["0xd85"]="cortex-x925"
    ["0xd44"]="cortex-x4"
    ["0xd4e"]="cortex-x3"
    ["0xd40"]="cortex-x2"
    ["0xd4b"]="cortex-x1"
    ["0xd4c"]="cortex-x1"
    ["0xd47"]="cortex-a720"  # A725 uses A720 tune
    ["0xd88"]="cortex-a720"
    ["0xd43"]="cortex-a715"
    ["0xd41"]="cortex-a78"
    ["0xd4d"]="cortex-a78"
    ["0xd0d"]="cortex-a77"
    ["0xd0b"]="cortex-a76"
    ["0xd0a"]="cortex-a75"
    ["0xd09"]="cortex-a73"
    ["0xd08"]="cortex-a72"
    ["0xd07"]="cortex-a57"
    ["0xd46"]="cortex-a520"
    ["0xd42"]="cortex-a510"
    ["0xd05"]="cortex-a55"
    ["0xd03"]="cortex-a53"
    ["0xd04"]="cortex-a35"
)

declare -A CPU_PART_TO_ARCH=(
    ["0xd85"]="armv9.2-a"
    ["0xd44"]="armv9.2-a"
    ["0xd4e"]="armv9-a"
    ["0xd40"]="armv9-a"
    ["0xd4b"]="armv8.2-a"
    ["0xd4c"]="armv8.2-a"
    ["0xd47"]="armv9.2-a"
    ["0xd88"]="armv9.2-a"
    ["0xd43"]="armv9-a"
    ["0xd41"]="armv8.2-a"
    ["0xd4d"]="armv8.2-a"
    ["0xd0d"]="armv8.2-a"
    ["0xd0b"]="armv8.2-a"
    ["0xd0a"]="armv8.2-a"
    ["0xd09"]="armv8-a"
    ["0xd08"]="armv8-a"
    ["0xd07"]="armv8-a"
    ["0xd46"]="armv9.2-a"
    ["0xd42"]="armv9-a"
    ["0xd05"]="armv8.2-a"
    ["0xd03"]="armv8-a"
    ["0xd04"]="armv8-a"
)

# ============================================================================
# Detection Functions
# ============================================================================

detect_cpu_parts() {
    # Get unique CPU part numbers
    grep "CPU part" /proc/cpuinfo 2>/dev/null | awk '{print $4}' | sort -u
}

detect_cpu_features() {
    # Get CPU features from /proc/cpuinfo
    grep -m1 "Features" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | tr ' ' '\n' | sort -u | tr '\n' ' '
}

has_feature() {
    local feature="$1"
    local features=$(detect_cpu_features)
    [[ " $features " == *" $feature "* ]]
}

detect_big_cores() {
    # Identify the prime/big core (highest part number usually)
    local parts=$(detect_cpu_parts)
    local biggest=""
    local biggest_name=""
    
    for part in $parts; do
        local name="${CPU_PART_TO_NAME[$part]:-unknown}"
        # X-cores and A7xx are big cores
        if [[ "$name" == *"X"* ]] || [[ "$name" == *"A7"* ]]; then
            if [[ -z "$biggest" ]] || [[ "$part" > "$biggest" ]]; then
                biggest="$part"
                biggest_name="$name"
            fi
        fi
    done
    
    echo "$biggest"
}

get_optimal_mcpu() {
    local big_core=$(detect_big_cores)
    
    if [[ -n "$big_core" ]] && [[ -n "${CPU_PART_TO_MCPU[$big_core]}" ]]; then
        echo "${CPU_PART_TO_MCPU[$big_core]}"
    else
        # Fallback to feature-based detection
        if has_feature "sve2"; then
            echo "cortex-a710"
        elif has_feature "sve"; then
            echo "cortex-a76"
        elif has_feature "asimddp"; then
            echo "cortex-a76"
        else
            echo "cortex-a55"
        fi
    fi
}

get_optimal_march() {
    local big_core=$(detect_big_cores)
    local base_arch=""
    
    if [[ -n "$big_core" ]] && [[ -n "${CPU_PART_TO_ARCH[$big_core]}" ]]; then
        base_arch="${CPU_PART_TO_ARCH[$big_core]}"
    else
        # Fallback to feature-based detection
        if has_feature "sve2"; then
            base_arch="armv9-a"
        elif has_feature "asimddp"; then
            base_arch="armv8.2-a"
        else
            base_arch="armv8-a"
        fi
    fi
    
    # Add extensions based on detected features
    local extensions=""
    
    if has_feature "sve2"; then
        extensions="${extensions}+sve2"
    elif has_feature "sve"; then
        extensions="${extensions}+sve"
    fi
    
    if has_feature "i8mm"; then
        extensions="${extensions}+i8mm"
    fi
    
    if has_feature "bf16"; then
        extensions="${extensions}+bf16"
    fi
    
    if has_feature "asimddp"; then
        extensions="${extensions}+dotprod"
    fi
    
    if has_feature "fphp" || has_feature "asimdhp"; then
        extensions="${extensions}+fp16"
    fi
    
    echo "${base_arch}${extensions}"
}

get_optimal_cflags() {
    local mcpu=$(get_optimal_mcpu)
    local march=$(get_optimal_march)
    
    echo "-O3 -mcpu=$mcpu -march=$march -flto"
}

get_memory_gb() {
    local mem_kb=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}')
    echo $((mem_kb / 1024 / 1024))
}

get_core_count() {
    nproc 2>/dev/null || grep -c "processor" /proc/cpuinfo 2>/dev/null || echo "4"
}

get_device_tier() {
    local ram_gb=$(get_memory_gb)
    
    if [[ $ram_gb -le 4 ]]; then
        echo "LOW"
    elif [[ $ram_gb -le 6 ]]; then
        echo "MEDIUM"
    elif [[ $ram_gb -le 8 ]]; then
        echo "HIGH"
    else
        echo "ULTRA"
    fi
}

# ============================================================================
# Output Functions
# ============================================================================

output_full() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║       ARM Device Detection for exo                       ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    echo -e "${YELLOW}=== CPU Information ===${NC}"
    echo ""
    
    # Detected cores
    echo -e "${BLUE}Detected Cores:${NC}"
    local parts=$(detect_cpu_parts)
    for part in $parts; do
        local name="${CPU_PART_TO_NAME[$part]:-Unknown}"
        local arch="${CPU_PART_TO_ARCH[$part]:-unknown}"
        echo "  $part: $name ($arch)"
    done
    echo ""
    
    # Core counts
    local total_cores=$(get_core_count)
    echo -e "${BLUE}Total Cores:${NC} $total_cores"
    
    # Big core
    local big_core=$(detect_big_cores)
    if [[ -n "$big_core" ]]; then
        echo -e "${BLUE}Primary (Big) Core:${NC} ${CPU_PART_TO_NAME[$big_core]:-unknown}"
    fi
    echo ""
    
    echo -e "${YELLOW}=== CPU Features ===${NC}"
    echo ""
    
    # Key features
    echo -e "${BLUE}Available Extensions:${NC}"
    has_feature "asimd" && echo -e "  ${GREEN}✓${NC} NEON/ASIMD (128-bit SIMD)"
    has_feature "asimddp" && echo -e "  ${GREEN}✓${NC} Dot Product (2-4x quantized speedup)"
    has_feature "fphp" && echo -e "  ${GREEN}✓${NC} FP16 (half precision)"
    has_feature "sve" && echo -e "  ${GREEN}✓${NC} SVE (scalable vectors)"
    has_feature "sve2" && echo -e "  ${GREEN}✓${NC} SVE2 (enhanced scalable vectors)"
    has_feature "i8mm" && echo -e "  ${GREEN}✓${NC} I8MM (int8 matrix multiply)"
    has_feature "bf16" && echo -e "  ${GREEN}✓${NC} BF16 (brain float)"
    has_feature "aes" && echo -e "  ${GREEN}✓${NC} AES (crypto acceleration)"
    has_feature "sha2" && echo -e "  ${GREEN}✓${NC} SHA2 (crypto acceleration)"
    echo ""
    
    echo -e "${YELLOW}=== Memory & Storage ===${NC}"
    echo ""
    local ram_gb=$(get_memory_gb)
    local tier=$(get_device_tier)
    echo -e "${BLUE}RAM:${NC} ${ram_gb}GB"
    echo -e "${BLUE}Device Tier:${NC} $tier"
    
    # Free storage
    local free_storage=$(df -BG ~ 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G')
    echo -e "${BLUE}Free Storage:${NC} ${free_storage}GB"
    echo ""
    
    echo -e "${YELLOW}=== Recommended Compiler Flags ===${NC}"
    echo ""
    local mcpu=$(get_optimal_mcpu)
    local march=$(get_optimal_march)
    local cflags=$(get_optimal_cflags)
    
    echo -e "${BLUE}-mcpu:${NC} $mcpu"
    echo -e "${BLUE}-march:${NC} $march"
    echo ""
    echo -e "${GREEN}Full CFLAGS:${NC}"
    echo "  $cflags"
    echo ""
    
    echo -e "${YELLOW}=== Model Recommendations ===${NC}"
    echo ""
    
    case $tier in
        LOW)
            echo "For ${ram_gb}GB RAM, recommended models:"
            echo -e "  ${GREEN}qwen-0.5b${NC}  - Ultra-light (400MB)"
            echo -e "  ${GREEN}tinyllama${NC}  - Good quality (700MB)"
            echo -e "  ${GREEN}llama-1b${NC}   - Best for tier (750MB)"
            ;;
        MEDIUM)
            echo "For ${ram_gb}GB RAM, recommended models:"
            echo -e "  ${GREEN}llama-3b${NC}   - Great balance (2GB)"
            echo -e "  ${GREEN}qwen-3b${NC}    - Strong reasoning (2GB)"
            echo -e "  ${GREEN}phi-3${NC}      - Best quality (2.3GB)"
            ;;
        HIGH)
            echo "For ${ram_gb}GB RAM, recommended models:"
            echo -e "  ${GREEN}phi-3${NC}      - Excellent (2.3GB)"
            echo -e "  ${GREEN}qwen-7b${NC}    - High quality (4GB)"
            echo -e "  ${GREEN}llama-8b${NC}   - Best single-device (4.5GB)"
            ;;
        ULTRA)
            echo "For ${ram_gb}GB RAM, recommended models:"
            echo -e "  ${GREEN}llama-8b${NC}   - Excellent quality (4.5GB)"
            echo -e "  ${GREEN}gemma-9b${NC}   - Google quality (5.5GB)"
            echo "Consider clustering for 13B+ models"
            ;;
    esac
    echo ""
    
    echo -e "${YELLOW}=== Performance Tips ===${NC}"
    echo ""
    echo "1. Use Q4_K_M quantization for best quality/speed balance"
    echo "2. Set thread count to big core count (usually 4)"
    echo "3. Keep device plugged in for sustained performance"
    echo "4. Monitor thermals with: ./scripts/thermal_monitor.sh"
    echo ""
}

output_flags() {
    get_optimal_cflags
}

output_json() {
    local mcpu=$(get_optimal_mcpu)
    local march=$(get_optimal_march)
    local cflags=$(get_optimal_cflags)
    local ram_gb=$(get_memory_gb)
    local tier=$(get_device_tier)
    local cores=$(get_core_count)
    local big_core=$(detect_big_cores)
    local big_name="${CPU_PART_TO_NAME[$big_core]:-unknown}"
    
    # Features as array
    local features=""
    has_feature "asimd" && features="${features}\"asimd\","
    has_feature "asimddp" && features="${features}\"dotprod\","
    has_feature "fphp" && features="${features}\"fp16\","
    has_feature "sve" && features="${features}\"sve\","
    has_feature "sve2" && features="${features}\"sve2\","
    has_feature "i8mm" && features="${features}\"i8mm\","
    has_feature "bf16" && features="${features}\"bf16\","
    features="${features%,}"  # Remove trailing comma
    
    cat << EOF
{
    "mcpu": "$mcpu",
    "march": "$march",
    "cflags": "$cflags",
    "ram_gb": $ram_gb,
    "tier": "$tier",
    "cores": $cores,
    "big_core": "$big_name",
    "features": [$features]
}
EOF
}

output_env() {
    local mcpu=$(get_optimal_mcpu)
    local march=$(get_optimal_march)
    local cflags=$(get_optimal_cflags)
    local ram_gb=$(get_memory_gb)
    local tier=$(get_device_tier)
    
    echo "export EXO_ARM_MCPU=\"$mcpu\""
    echo "export EXO_ARM_MARCH=\"$march\""
    echo "export EXO_ARM_CFLAGS=\"$cflags\""
    echo "export EXO_RAM_GB=\"$ram_gb\""
    echo "export EXO_DEVICE_TIER=\"$tier\""
    echo "export CFLAGS=\"$cflags\""
    echo "export CXXFLAGS=\"$cflags\""
}

# ============================================================================
# Main
# ============================================================================

case "$MODE" in
    "--flags"|"-f")
        output_flags
        ;;
    "--json"|"-j")
        output_json
        ;;
    "--env"|"-e")
        output_env
        ;;
    "--help"|"-h")
        echo "ARM Device Detection for exo"
        echo ""
        echo "Usage: $0 [option]"
        echo ""
        echo "Options:"
        echo "  (none)       Full detection report"
        echo "  --flags, -f  Output only compiler flags"
        echo "  --json, -j   Output as JSON"
        echo "  --env, -e    Output as environment variables (use with 'source')"
        echo "  --help, -h   Show this help"
        echo ""
        echo "Examples:"
        echo "  ./arm_detect.sh                    # Full report"
        echo "  ./arm_detect.sh --flags            # Just CFLAGS"
        echo "  source ./arm_detect.sh --env       # Set environment variables"
        echo ""
        ;;
    *)
        output_full
        ;;
esac

