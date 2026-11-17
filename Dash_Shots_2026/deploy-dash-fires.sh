#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="/opt/openshift/applications/ai-lab/dash-fires/Dash_Shots_2026"
NAMESPACE="ai-lab"
APP="python-dash-shots"
TAG="$(date -u +%Y%m%d%H%M%S)"
BUILD_DIR=""

cleanup() {
    if [[ -n "$BUILD_DIR" && -d "$BUILD_DIR" ]]; then
        rm -rf "$BUILD_DIR"
    fi
}
trap cleanup EXIT

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

echo "=============================================="
echo " DASH FIRES - OPENSHIFT DEPLOYMENT"
echo "=============================================="

# STEP 1 - PREFLIGHT CHECKS
echo
echo "[1/8] Checking environment..."

for tool in oc python3 tar; do
    command -v "$tool" >/dev/null || fail "Missing $tool"
done

oc whoami >/dev/null || fail "Log into OpenShift first"

[[ -f "$APP_DIR/app.py" ]] || fail "Missing app.py"
[[ -f "$APP_DIR/Dockerfile" ]] || fail "Missing Dockerfile"
[[ -f "$APP_DIR/requirements.txt" ]] || fail "Missing requirements.txt"
[[ -d "$APP_DIR/SubApps" ]] || fail "Missing SubApps directory"
[[ -d "$APP_DIR/data" ]] || fail "Missing data directory"

if ! oc get namespace "$NAMESPACE" >/dev/null 2>&1; then
    fail "Namespace $NAMESPACE does not exist. Create it first or change NAMESPACE."
fi

for permission in \
    "create builds" \
    "create buildconfigs" \
    "create imagestreams" \
    "create deployments" \
    "create services" \
    "create routes"; do

    read -r verb resource <<< "$permission"

    [[ "$(oc auth can-i "$verb" "$resource" -n "$NAMESPACE")" == "yes" ]] \
        || fail "Missing permission: $permission in $NAMESPACE"
done

echo "OpenShift connection: OK"
echo "Namespace: $NAMESPACE"

# STEP 2 - PREPARE BUILD DIRECTORY
echo
echo "[2/8] Preparing application source..."

BUILD_DIR="$(mktemp -d)"

cp -a "$APP_DIR/." "$BUILD_DIR/"

rm -rf \
    "$BUILD_DIR/.git" \
    "$BUILD_DIR/.venv" \
    "$BUILD_DIR/venv" \
    "$BUILD_DIR/__pycache__"

find "$BUILD_DIR" -type d -name __pycache__ \
    -prune -exec rm -rf {} +

# STEP 3 - CORRECT PYTHON APPLICATION
echo
echo "[3/8] Validating Dash and dependencies..."

python3 - "$BUILD_DIR" <<'PY'
from pathlib import Path
import ast
import sys

p = Path(sys.argv[1])
app = p / "app.py"
src = app.read_text()

tree = ast.parse(src)

has_server = False

for node in tree.body:
    if isinstance(node, ast.Assign):
        if any(
            isinstance(t, ast.Name) and t.id == "server"
            for t in node.targets
        ):
            has_server = True

    elif isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            if node.target.id == "server":
                has_server = True

if not has_server:
    needle = 'if __name__ == "__main__":'

    if needle not in src:
        raise SystemExit(
            "Cannot locate main guard in app.py"
        )

    src = src.replace(
        needle,
        "server = app.server\n\n" + needle,
        1
    )

    app.write_text(src)
    print("Added Gunicorn server entry point.")

# Clean requirements
req = p / "requirements.txt"

lines = req.read_text().splitlines()

clean = []
seen = set()

for line in lines:
    value = line.strip()

    if not value or value.startswith("#"):
        continue

    key = value.lower().replace("_", "-")

    if key in ("plotly.express", "pyephem", "pip"):
        continue

    if key in seen:
        continue

    seen.add(key)
    clean.append(value)

req.write_text("\n".join(clean) + "\n")

print("Python requirements prepared.")

# Check Docker ignore
ignore = p / ".dockerignore"

if ignore.exists():
    patterns = ignore.read_text().splitlines()

    for entry in ("data", "SubApps", "Icons"):
        if any(
            x.strip().strip("/") in (entry, "*")
            for x in patterns
            if not x.strip().startswith("#")
        ):
            print(
                f"WARNING: .dockerignore may exclude {entry}"
            )
PY


# DASH_FIRES_DOCKERFILE_FIX
echo "Correcting Dockerfile for OpenShift non-root builds..."

python3 - "$BUILD_DIR" <<'DOCKERFIX'
from pathlib import Path
import sys

root = Path(sys.argv[1])
dockerfile = root / "Dockerfile"
src = dockerfile.read_text()

old = (
    "RUN chgrp -R 0 "
    "/opt/openshift/applications/ai-lab/dash-fires "
    "&& chmod -R g=u "
    "/opt/openshift/applications/ai-lab/dash-fires"
)

if old not in src:
    raise SystemExit(
        "Expected Dockerfile permission command not found. "
        "Stopping instead of modifying an unknown Dockerfile."
    )

# UBI Python runs as a non-root user. Switch to root
# only for image file permissions, then restore UID 1001.
new = (
    "USER 0\n"
    + old
    + "\nUSER 1001"
)

src = src.replace(old, new, 1)
dockerfile.write_text(src)

# Exclude build artifacts and deployment scripts
ignore = root / ".dockerignore"

existing = ignore.read_text() if ignore.exists() else ""

extra = """
__pycache__/
*.pyc
.git/
.venv/
venv/
deploy-dash-fires.sh
deploy-dash-fires.sh.bak.*
deploy-openshift.ps1
"""

ignore.write_text(existing + "\n" + extra)

print("Dockerfile corrected.")
print("Build exclusions updated.")
DOCKERFIX

python3 -m py_compile "$BUILD_DIR/app.py"

for f in "$BUILD_DIR"/SubApps/*.py; do
    python3 -m py_compile "$f"
done

echo "Python syntax validation: OK"

# STEP 4 - CREATE IMAGESTREAM
echo
echo "[4/8] Configuring OpenShift ImageStream..."

oc -n "$NAMESPACE" create imagestream "$APP" \
    --dry-run=client -o yaml \
    | oc -n "$NAMESPACE" apply -f -

# STEP 5 - CREATE BUILD CONFIGURATION
echo
echo "[5/8] Configuring OpenShift build..."

if ! oc -n "$NAMESPACE" get bc "$APP" \
    >/dev/null 2>&1; then

    oc -n "$NAMESPACE" new-build \
        --name="$APP" \
        --binary \
        --strategy=docker \
        --to="$APP:latest"
fi

SOURCE_TYPE="$(
    oc -n "$NAMESPACE" get bc "$APP" \
        -o jsonpath='{.spec.source.type}'
)"

STRATEGY_TYPE="$(
    oc -n "$NAMESPACE" get bc "$APP" \
        -o jsonpath='{.spec.strategy.type}'
)"

[[ "$SOURCE_TYPE" == "Binary" ]] \
    || fail "Existing BuildConfig is not binary source"

[[ "$STRATEGY_TYPE" == "Docker" ]] \
    || fail "Existing BuildConfig is not Docker strategy"

# STEP 6 - BUILD CONTAINER
echo
echo "[6/8] Building Dash Fires container..."

oc -n "$NAMESPACE" start-build "$APP" \
    --from-dir="$BUILD_DIR" \
    --follow \
    --wait

IMAGE="$(
    oc -n "$NAMESPACE" get istag "$APP:latest" \
        -o jsonpath='{.image.dockerImageReference}'
)"

[[ -n "$IMAGE" ]] \
    || fail "Build produced no image reference"

echo "Container image: $IMAGE"

# STEP 7 - DEPLOY APPLICATION
echo
echo "[7/8] Deploying Dash Fires..."

cat <<YAML | oc -n "$NAMESPACE" apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $APP
  labels:
    app: $APP
spec:
  replicas: 1
  selector:
    matchLabels:
      app: $APP
  template:
    metadata:
      labels:
        app: $APP
      annotations:
        dash-fires/build-tag: "$TAG"
    spec:
      containers:
        - name: app
          image: "$IMAGE"
          imagePullPolicy: Always
          ports:
            - name: http
              containerPort: 8080
          env:
            - name: PORT
              value: "8080"
            - name: GOOGLE_MAPS_API_KEY
              valueFrom:
                secretKeyRef:
                  name: google-geo
                  key: GOOGLE_MAPS_API_KEY
          resources:
            requests:
              cpu: 100m
              memory: 512Mi
            limits:
              cpu: "2"
              memory: 2Gi
          readinessProbe:
            httpGet:
              path: /
              port: http
            initialDelaySeconds: 15
            periodSeconds: 10
            failureThreshold: 12
          livenessProbe:
            httpGet:
              path: /
              port: http
            initialDelaySeconds: 60
            periodSeconds: 20
          securityContext:
            allowPrivilegeEscalation: false
            runAsNonRoot: true
            capabilities:
              drop:
                - ALL
            seccompProfile:
              type: RuntimeDefault
---
apiVersion: v1
kind: Service
metadata:
  name: $APP
spec:
  selector:
    app: $APP
  ports:
    - name: http
      port: 8080
      targetPort: http
---
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: $APP
spec:
  to:
    kind: Service
    name: $APP
  port:
    targetPort: http
  tls:
    termination: edge
    insecureEdgeTerminationPolicy: Redirect
YAML

# STEP 8 - VERIFY DEPLOYMENT
echo
echo "[8/8] Verifying application rollout..."

if ! oc -n "$NAMESPACE" rollout status \
    "deployment/$APP" --timeout=8m; then

    echo
    echo "===== POD STATUS ====="
    oc -n "$NAMESPACE" get pods \
        -l "app=$APP" -o wide || true

    echo
    echo "===== DEPLOYMENT DETAILS ====="
    oc -n "$NAMESPACE" describe deployment "$APP" || true

    echo
    echo "===== APPLICATION LOGS ====="
    oc -n "$NAMESPACE" logs \
        "deployment/$APP" \
        --all-containers --tail=100 || true

    fail "Deployment rollout failed"
fi

HOST="$(
    oc -n "$NAMESPACE" get route "$APP" \
        -o jsonpath='{.spec.host}'
)"

echo
echo "=============================================="
echo " DASH FIRES DEPLOYMENT COMPLETE"
echo "=============================================="
echo
echo "Namespace: $NAMESPACE"
echo "Application: $APP"
echo "Dashboard: https://$HOST"
echo
echo "View pods:"
echo "oc -n $NAMESPACE get pods"
echo
echo "View logs:"
echo "oc -n $NAMESPACE logs -f deployment/$APP"
echo
echo "View route:"
echo "oc -n $NAMESPACE get route $APP"
echo
echo "NOTE: HTTPS does not authenticate users."
echo "Apply access controls for sensitive data."
echo
