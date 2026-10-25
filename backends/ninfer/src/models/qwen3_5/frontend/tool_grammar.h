#pragma once
#include "models/qwen3_5/frontend/tool_call_parser.h"
namespace ninfer::models::qwen3_5::frontend {
std::string tool_grammar(const ToolCallOutputContract&, const StructuredOutputOptions&);
}
