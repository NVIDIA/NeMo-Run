---
description: "Administrative guide for NeMo Run including deployment, maintenance, version management, and operational procedures."
tags: ["administration", "deployment", "maintenance", "configuration", "monitoring", "troubleshooting"]
categories: ["guides"]
---

(admin)=

# NeMo Run Administration

This guide covers administrative tasks for NeMo Run including deployment, maintenance, version management, and operational procedures. This is essential reading for system administrators, DevOps engineers, and anyone responsible for managing NeMo Run installations.

## Overview

NeMo Run administration encompasses several key areas:

- **Deployment Management**: Installing and configuring NeMo Run across different environments
- **Version Management**: Upgrading, downgrading, and managing NeMo Run versions
- **Configuration Management**: Setting up environment-specific configurations
- **Monitoring and Logging**: Implementing monitoring and log management
- **Security**: Managing authentication, authorization, and security policies
- **Performance Optimization**: Tuning for optimal performance and resource utilization
- **Backup and Recovery**: Implementing backup strategies and disaster recovery procedures

## System Requirements

Before deploying NeMo Run, ensure your system meets these requirements:

### Hardware Requirements

- **CPU**: Minimum 4 cores, recommended 8+ cores
- **Memory**: Minimum 8GB RAM, recommended 16GB+ RAM
- **Storage**: Minimum 10GB free space, recommended 50GB+ for experiment data
- **Network**: Stable internet connection for package installation and updates

### Software Requirements

- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+), macOS 10.15+, or Windows 10+ with WSL2
- **Python**: 3.8 or higher
- **pip**: Latest version
- **Git**: For source installation and version control
- **Docker**: For containerized execution (optional but recommended)

### Network Requirements

- **Internet Access**: For package installation and updates
- **SSH Access**: For remote cluster management
- **Firewall Configuration**: Open ports for cluster communication (if applicable)

## Installation and Deployment

### Standard Installation

Install NeMo Run using the standard method:

```bash
# Install NeMo Run
pip install git+https://github.com/NVIDIA-NeMo/Run.git

# Verify installation
python -c "import nemo_run; print(nemo_run.__version__)"
```

### Source Installation

For development or custom modifications:

```bash
# Clone the repository
git clone https://github.com/nvidia/nemo-run.git
cd nemo-run

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Containerized Deployment

Deploy NeMo Run using Docker:

```bash
# Pull the official image
docker pull nvcr.io/nvidia/nemo-run:latest

# Run NeMo Run in a container
docker run --rm -it nvcr.io/nvidia/nemo-run:latest
```

### Environment Configuration

Set up environment variables for NeMo Run:

```bash
# NeMo Run home directory
export NEMORUN_HOME=~/.nemo_run

# Logging configuration
export NEMORUN_LOG_LEVEL=INFO
export NEMORUN_LOG_FILE=/var/log/nemo-run.log

# Execution settings
export NEMORUN_DEFAULT_EXECUTOR=local
export NEMORUN_MAX_CONCURRENT_JOBS=10

# Security settings
export NEMORUN_SKIP_CONFIRMATION=false
export NEMORUN_VERBOSE_LOGGING=false
```

## Version Management

### Upgrading NeMo Run

Upgrade to the latest version:

```bash
# Upgrade to latest version
pip install --upgrade git+https://github.com/NVIDIA-NeMo/Run.git

# Verify upgrade
python -c "import nemo_run; print(nemo_run.__version__)"
```

### Upgrading with Dependencies

Upgrade with all optional dependencies:

```bash
# Upgrade with all optional dependencies
pip install --upgrade git+https://github.com/NVIDIA-NeMo/Run.git[all]

# Or upgrade specific optional dependencies
pip install git+https://github.com/NVIDIA-NeMo/Run.git[skypilot]
pip install git+https://github.com/NVIDIA-NeMo/Run.git[ray]
```

### Downgrading NeMo Run

Downgrade to a specific version:

```bash
# Downgrade to specific version
pip install git+https://github.com/NVIDIA-NeMo/Run.git@v0.2.0

# Or install specific commit
pip install git+https://github.com/NVIDIA-NeMo/Run.git@commit-hash
```

### Version Compatibility

Check version compatibility with dependencies:

```bash
# Check installed versions
pip list | grep nemo-run
pip list | grep torchx
pip list | grep fiddle

# Check for version conflicts
pip check
```

## Configuration Management

### Environment-Specific Configurations

Set up different configurations for different environments:

#### Development Environment

```bash
# Development settings
export NEMORUN_ENV=development
export NEMORUN_LOG_LEVEL=DEBUG
export NEMORUN_DEFAULT_EXECUTOR=local
```

#### Staging Environment

```bash
# Staging settings
export NEMORUN_ENV=staging
export NEMORUN_LOG_LEVEL=INFO
export NEMORUN_DEFAULT_EXECUTOR=docker
```

#### Production Environment

```bash
# Production settings
export NEMORUN_ENV=production
export NEMORUN_LOG_LEVEL=WARNING
export NEMORUN_DEFAULT_EXECUTOR=slurm
```

### Configuration Files

Create configuration files for different environments:

```bash
# Create configuration directory
mkdir -p ~/.config/nemo-run

# Development configuration
cat > ~/.config/nemo-run/development.yaml << EOF
execution:
  default_backend: local
  max_concurrent_jobs: 5
  timeout: 1800

logging:
  level: DEBUG
  file: /tmp/nemo-run-dev.log

security:
  skip_confirmation: true
  verbose_logging: true
EOF

# Production configuration
cat > ~/.config/nemo-run/production.yaml << EOF
execution:
  default_backend: slurm
  max_concurrent_jobs: 50
  timeout: 86400

logging:
  level: WARNING
  file: /var/log/nemo-run.log

security:
  skip_confirmation: false
  verbose_logging: false
EOF
```

## Monitoring and Logging

### Log Management

Configure comprehensive logging:

```bash
# Set up log rotation
sudo logrotate -f /etc/logrotate.d/nemo-run

# Monitor logs in real-time
tail -f /var/log/nemo-run.log

# Search logs for errors
grep -i error /var/log/nemo-run.log
```

### System Monitoring

Monitor system resources and NeMo Run performance:

```bash
# Monitor disk usage
df -h ~/.nemo_run

# Monitor memory usage
free -h

# Monitor CPU usage
top -p $(pgrep -f nemo-run)

# Monitor network connections
netstat -tulpn | grep nemo-run
```

### Health Checks

Implement health checks for NeMo Run services:

```bash
# Check NeMo Run status
python -c "import nemo_run; print('NeMo Run is healthy')"

# Check CLI availability
python -c "from nemo_run.__main__ import app; print('CLI is available')"

# Check executor availability
python -c "from nemo_run.core.execution import LocalExecutor; print('Executors are available')"
```

## Security Management

### Authentication and Authorization

Configure security settings:

```bash
# Enable authentication
export NEMORUN_REQUIRE_AUTH=true
export NEMORUN_AUTH_TOKEN=your-secure-token

# Set up user permissions
chmod 600 ~/.nemo_run/config.yaml
chown $USER:$USER ~/.nemo_run
```

### Network Security

Configure network security for cluster environments:

```bash
# Configure firewall rules
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 8080/tcp  # Web interface (if applicable)

# Set up VPN access for remote clusters
# Configure SSH key-based authentication
ssh-keygen -t rsa -b 4096 -C "nemo-run-admin"
```

### Data Security

Implement data security measures:

```bash
# Encrypt sensitive data
gpg --encrypt --recipient admin@company.com config.yaml

# Set up secure backups
tar -czf nemo-run-backup-$(date +%Y%m%d).tar.gz ~/.nemo_run
gpg --encrypt --recipient admin@company.com nemo-run-backup-$(date +%Y%m%d).tar.gz
```

## Backup and Recovery

### Backup Procedures

Implement regular backup procedures:

```bash
# Create backup directory
mkdir -p ~/nemo-run-backups

# Create daily backup
cp -r ~/.nemo_run ~/nemo-run-backups/nemo-run-$(date +%Y%m%d)

# Create compressed backup
tar -czf ~/nemo-run-backups/nemo-run-$(date +%Y%m%d).tar.gz ~/.nemo_run

# Clean up old backups (keep last 30 days)
find ~/nemo-run-backups -name "nemo-run-*" -mtime +30 -delete
```

### Recovery Procedures

Implement recovery procedures:

```bash
# Stop NeMo Run services
pkill -f nemo-run

# Restore from backup
rm -rf ~/.nemo_run
cp -r ~/nemo-run-backups/nemo-run-20231201 ~/.nemo_run

# Verify restoration
python -c "import nemo_run; print('Recovery successful')"
```

### Disaster Recovery

Plan for disaster recovery scenarios:

```bash
# Create disaster recovery script
cat > disaster-recovery.sh << 'EOF'
#!/bin/bash

# Stop all NeMo Run processes
pkill -f nemo-run

# Restore from latest backup
LATEST_BACKUP=$(ls -t ~/nemo-run-backups/nemo-run-*.tar.gz | head -1)
tar -xzf $LATEST_BACKUP -C ~/

# Verify restoration
python -c "import nemo_run; print('Disaster recovery completed')"
EOF

chmod +x disaster-recovery.sh
```

## Performance Optimization

### Resource Management

Optimize resource usage:

```bash
# Set memory limits
export NEMORUN_MAX_MEMORY=8GB
export NEMORUN_MEMORY_POOL_SIZE=2GB

# Set CPU limits
export NEMORUN_MAX_CPUS=8
export NEMORUN_CPU_AFFINITY=true

# Configure caching
export NEMORUN_CACHE_DIR=/tmp/nemo-run-cache
export NEMORUN_CACHE_SIZE=5GB
```

### Network Optimization

Optimize network performance:

```bash
# Configure network timeouts
export NEMORUN_NETWORK_TIMEOUT=30
export NEMORUN_MAX_CONNECTIONS=100
export NEMORUN_KEEPALIVE=true

# Enable compression
export NEMORUN_COMPRESSION=true
export NEMORUN_CHUNK_SIZE=1MB
```

### Storage Optimization

Optimize storage usage:

```bash
# Clean up old experiment data
find ~/.nemo_run/experiments -mtime +90 -delete

# Compress old logs
find ~/.nemo_run/logs -name "*.log" -mtime +30 -exec gzip {} \;

# Monitor disk usage
du -sh ~/.nemo_run
```

## Troubleshooting

### Common Issues

#### Installation Issues

```bash
# Check Python version
python --version

# Check pip version
pip --version

# Reinstall with clean environment
pip uninstall nemo-run -y
pip install git+https://github.com/NVIDIA-NeMo/Run.git
```

#### Permission Issues

```bash
# Fix permission issues
sudo chown -R $USER:$USER ~/.nemo_run
chmod -R 755 ~/.nemo_run

# Check file permissions
ls -la ~/.nemo_run
```

#### Network Issues

```bash
# Test network connectivity
ping github.com

# Test SSH connectivity
ssh -T git@github.com

# Check firewall settings
sudo ufw status
```

### Debug Mode

Enable debug mode for troubleshooting:

```bash
# Enable debug logging
export NEMORUN_DEBUG=true
export NEMORUN_LOG_LEVEL=DEBUG

# Run with verbose output
python -c "import nemo_run; print('Debug mode enabled')"
```

### Diagnostic Commands

Run diagnostic commands to identify issues:

```bash
# System diagnostics
python -c "import nemo_run; print(nemo_run.__version__)"
echo $NEMORUN_HOME
ls ~/.nemo_run/experiments/

# Network diagnostics
curl -I https://github.com
ssh -T git@github.com

# Resource diagnostics
df -h
free -h
top -n 1
```

## Maintenance Procedures

### Regular Maintenance

Schedule regular maintenance tasks:

```bash
# Daily maintenance
find ~/.nemo_run/logs -name "*.log" -mtime +7 -delete

# Weekly maintenance
find ~/.nemo_run/experiments -mtime +30 -delete
pip list --outdated

# Monthly maintenance
pip install --upgrade git+https://github.com/NVIDIA-NeMo/Run.git
tar -czf ~/nemo-run-backups/monthly-$(date +%Y%m).tar.gz ~/.nemo_run
```

### Update Procedures

Follow proper update procedures:

```bash
# Create backup before update
cp -r ~/.nemo_run ~/.nemo_run.backup.$(date +%Y%m%d)

# Update NeMo Run
pip install --upgrade git+https://github.com/NVIDIA-NeMo/Run.git

# Verify update
python -c "import nemo_run; print(f'Updated to version: {nemo_run.__version__}')"

# Test functionality
python -c "from nemo_run.core.execution import LocalExecutor; print('Update successful')"
```

### Cleanup Procedures

Implement cleanup procedures:

```bash
# Remove old experiments
find ~/.nemo_run/experiments -mtime +90 -exec rm -rf {} \;

# Remove old logs
find ~/.nemo_run/logs -name "*.log" -mtime +30 -delete

# Remove old cache
find /tmp/nemo-run-cache -mtime +7 -delete

# Remove old backups
find ~/nemo-run-backups -name "*.tar.gz" -mtime +365 -delete
```

This administrative guide provides comprehensive coverage of NeMo Run administration tasks. For specific deployment scenarios or advanced configurations, refer to the [Configuration Guide](configuration.md) and [Execution Guide](execution.md).
