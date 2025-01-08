# GPU Mode NVIDIA ARC Setup with Actions Runner Controller

This guide walks through setup of the Actions Runner Controller (ARC) for GitHub Actions, including configuring GPU resources for your runners using Kubernetes and Kyverno.

---

## Prerequisites

1. **Kubernetes Cluster**: A working Kubernetes cluster.
2. **NVIDIA GPU Operator**: A working deployment of the NVIDIA GPU Operator
3. **kubectl**: Installed and configured for your cluster.
4. **Helm**: Installed and up-to-date.
5. **GitHub Personal Access Token (PAT)**: Required for authenticating with GitHub. Ensure it has the necessary permissions.

---

## Installation Steps

### 1. Clone the Actions Runner Controller Repository
```bash
git clone https://github.com/actions/actions-runner-controller.git
```

### 2. Install Local-Path Provisioner
The local-path provisioner is required for creating Persistent Volume Claims (PVCs) for pods.
```bash
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/master/deploy/local-path-storage.yaml
```

### 3. Install Kyverno
Kyverno is used for mutating admission webhooks to handle GPU-specific resource assignments.
```bash
helm repo add kyverno https://kyverno.github.io/kyverno/
helm repo update
helm install kyverno kyverno/kyverno -n kyverno --create-namespace
```

### 4. Install Runner Scale Set Controller
```bash
NAMESPACE="arc-systems"
helm install arc \
    --namespace "${NAMESPACE}" \
    --create-namespace \
    oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller
```

### 5. Install and Configure Runner Set
1. Navigate to the chart directory:
   ```bash
   cd ~/actions-runner-controller/charts/gha-runner-scale-set
   ```

2. Set up your environment variables:
   ```bash
   INSTALLATION_NAME="gpumode-nvidia-arc"
   NAMESPACE="arc-runners"
   GITHUB_CONFIG_URL="https://github.com/gpu-mode"  # Change this for repo-specific setup.
   GITHUB_PAT="ghp_******"                          # Replace with your actual PAT.
   ```

3. Install the runner set:
   ```bash
   helm install "${INSTALLATION_NAME}" \
       --namespace "${NAMESPACE}" \
       --create-namespace \
       --set githubConfigUrl="${GITHUB_CONFIG_URL}" \
       --set githubConfigSecret.github_token="${GITHUB_PAT}" \
       --set containerMode.type="kubernetes" \
       .
   ```

### 6. Configure GPU Mutating Admission Webhooks
Apply the following Kyverno policy to inject GPU resource requests and tolerations into pod. This configuration ensures a single GPU is allocated to pods scheduled inside the target `NAMESPACE`:
```bash
NAMESPACE="arc-runners"
cat <<EOF | kubectl apply -f -
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: inject-gpu-resources-and-tolerations
spec:
  rules:
    - name: add-gpu-resources-tolerations-and-nodeselector
      match:
        resources:
          kinds:
            - Pod
          namespaces:
            - ${NAMESPACE}
      mutate:
        patchStrategicMerge:
          spec:
            containers:
              - (name): "*"
                resources:
                  requests:
                    nvidia.com/gpu: "1"
                  limits:
                    nvidia.com/gpu: "1"
                securityContext:
                  capabilities:
                    add:
                      - SYS_ADMIN
                      - SYS_RESOURCE
                      - SYS_PTRACE
            tolerations:
              - key: "nvidia.com/gpu"
                operator: "Exists"
                effect: "NoSchedule"
            nodeSelector:
              nvidia.com/gpu.present: "true"
EOF
```

### 7. Set Default Limits/Requests for Namespace
Define default GPU limits and requests for the `arc-runners` namespace:
```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: LimitRange
metadata:
  name: gpu-limits
  namespace: arc-runners
spec:
  limits:
    - default:
        nvidia.com/gpu: "1"
      defaultRequest:
        nvidia.com/gpu: "1"
      type: Container
EOF
```

### 8. Validate GPU Mutation Policy
Create a test pod to validate the mutation policy:
```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test-pod
  namespace: arc-runners
spec:
  containers:
    - name: gpu-test-container
      image: nvidia/cuda:12.4.0-devel-ubuntu22.04
      command: ["sh", "-c", "echo 'Testing GPU mutation policy with CUDA 12.4 and Ubuntu 22.04' && sleep 3600"]
EOF
```

Inspect the pod and confirm GPU resources have been assigned. Clean up the test pod:
```bash
kubectl delete pod -n arc-runners gpu-test-pod
```

---

## Helm Commands for ARC Management

### Uninstall ARC
```bash
helm uninstall "${INSTALLATION_NAME}" \
    --namespace "${NAMESPACE}"
```

### Upgrade ARC Configuration
Modify the `values.yaml` and upgrade the deployment:
```bash
helm upgrade "${INSTALLATION_NAME}" \
    --namespace "${NAMESPACE}" \
    --set githubConfigUrl="${GITHUB_CONFIG_URL}" \
    --set githubConfigSecret.github_token="${GITHUB_PAT}" \
    --set containerMode.type="kubernetes" \
    .
```

---

## Notes

- **Pod Cleanup**: After updating the Kyverno policy, restart pods to apply changes:
  ```bash
  kubectl get pods -n arc-runners | awk '{if(NR>1)print $1}' | xargs -r kubectl delete pod -n arc-runners
  ```

- **GPU Configuration**: Ensure your nodes have the label `nvidia.com/gpu.present=true` to match the node selector.

---

## Troubleshooting

- Check pod events and logs to debug any issues:
  ```bash
  kubectl describe pod <pod-name> -n <namespace>
  kubectl logs <pod-name> -n <namespace>
  ```
