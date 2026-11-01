#pragma once

// All value types of the in-tree Engine interface. Only ninfer/engine.h includes this umbrella;
// other code includes the header that defines the types it uses, so a change to one domain does
// not recompile the code of the others (scripts/check-include-boundaries.py).

#include "ninfer/base.h"
#include "ninfer/engine_options.h"
#include "ninfer/kv_cache_storage.h"
#include "ninfer/model_enums.h"
#include "ninfer/observability/execution.h"
#include "ninfer/observability/resources.h"
#include "ninfer/observability/snapshot.h"
#include "ninfer/request.h"
#include "ninfer/startup.h"
#include "ninfer/types_fwd.h"
