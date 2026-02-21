//! Access Control Module
//!
//! Permission-based access control for ALICE-Zip archives.
//!
//! # Features
//!
//! - File-level permissions (read, write, delete)
//! - Role-based access control
//! - Access tokens for temporary access
//!
//! # Usage
//!
//! ```rust,ignore
//! use alice_enterprise::access::{AccessControl, Permission, Role};
//!
//! let mut acl = AccessControl::new();
//!
//! // Grant read access
//! acl.grant("user123", Permission::READ);
//!
//! // Check permission
//! if acl.check("user123", Permission::READ) {
//!     // Allow access
//! }
//! ```

use std::collections::HashMap;
use thiserror::Error;
use bitflags::bitflags;

/// Access control errors
#[derive(Error, Debug)]
pub enum AccessError {
    /// Permission denied
    #[error("permission denied: {0}")]
    PermissionDenied(String),

    /// User not found
    #[error("user not found: {0}")]
    UserNotFound(String),

    /// Invalid token
    #[error("invalid access token")]
    InvalidToken,

    /// Token expired
    #[error("access token expired")]
    TokenExpired,
}

bitflags! {
    /// Permission flags
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Permission: u8 {
        /// Read files from archive
        const READ = 0x01;
        /// Add files to archive
        const WRITE = 0x02;
        /// Delete files from archive
        const DELETE = 0x04;
        /// Modify archive metadata
        const MODIFY = 0x08;
        /// Manage permissions
        const ADMIN = 0x10;

        /// Full access
        const FULL = Self::READ.bits() | Self::WRITE.bits() |
                     Self::DELETE.bits() | Self::MODIFY.bits() | Self::ADMIN.bits();
    }
}

/// User roles
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Read-only access
    Viewer,
    /// Read and write access
    Editor,
    /// Full access except admin
    Contributor,
    /// Full access including admin
    Admin,
    /// Owner (cannot be revoked)
    Owner,
}

impl Role {
    /// Get default permissions for role
    pub fn permissions(&self) -> Permission {
        match self {
            Self::Viewer => Permission::READ,
            Self::Editor => Permission::READ | Permission::WRITE,
            Self::Contributor => Permission::READ | Permission::WRITE | Permission::DELETE | Permission::MODIFY,
            Self::Admin | Self::Owner => Permission::FULL,
        }
    }
}

/// Access control entry
#[derive(Debug, Clone)]
pub struct AccessEntry {
    /// User or group ID
    pub subject: String,
    /// Assigned role
    pub role: Role,
    /// Additional permissions (beyond role defaults)
    pub extra_permissions: Permission,
    /// Denied permissions (override role defaults)
    pub denied_permissions: Permission,
}

impl AccessEntry {
    /// Create new entry with role
    pub fn new(subject: impl Into<String>, role: Role) -> Self {
        Self {
            subject: subject.into(),
            role,
            extra_permissions: Permission::empty(),
            denied_permissions: Permission::empty(),
        }
    }

    /// Get effective permissions
    pub fn effective_permissions(&self) -> Permission {
        (self.role.permissions() | self.extra_permissions) - self.denied_permissions
    }

    /// Check if permission is granted
    pub fn has_permission(&self, permission: Permission) -> bool {
        self.effective_permissions().contains(permission)
    }
}

/// Access control list for an archive
#[derive(Debug, Default)]
pub struct AccessControl {
    /// Owner ID
    owner: Option<String>,
    /// Access entries by subject
    entries: HashMap<String, AccessEntry>,
}

impl AccessControl {
    /// Create new empty ACL
    pub fn new() -> Self {
        Self::default()
    }

    /// Create ACL with owner
    pub fn with_owner(owner: impl Into<String>) -> Self {
        let owner_id = owner.into();
        let mut acl = Self::new();
        acl.owner = Some(owner_id.clone());
        acl.entries.insert(owner_id.clone(), AccessEntry::new(owner_id, Role::Owner));
        acl
    }

    /// Set owner
    pub fn set_owner(&mut self, owner: impl Into<String>) {
        let owner_id = owner.into();
        self.owner = Some(owner_id.clone());
        self.entries.insert(owner_id.clone(), AccessEntry::new(owner_id, Role::Owner));
    }

    /// Grant role to subject
    pub fn grant_role(&mut self, subject: impl Into<String>, role: Role) {
        let subject_id = subject.into();
        self.entries.insert(subject_id.clone(), AccessEntry::new(subject_id, role));
    }

    /// Grant specific permission
    pub fn grant_permission(&mut self, subject: impl Into<String>, permission: Permission) {
        let subject_id = subject.into();
        self.entries
            .entry(subject_id.clone())
            .or_insert_with(|| AccessEntry::new(subject_id, Role::Viewer))
            .extra_permissions |= permission;
    }

    /// Deny specific permission
    pub fn deny_permission(&mut self, subject: impl Into<String>, permission: Permission) {
        let subject_id = subject.into();
        self.entries
            .entry(subject_id.clone())
            .or_insert_with(|| AccessEntry::new(subject_id, Role::Viewer))
            .denied_permissions |= permission;
    }

    /// Revoke all access for subject
    pub fn revoke(&mut self, subject: &str) -> Result<(), AccessError> {
        // Cannot revoke owner
        if self.owner.as_deref() == Some(subject) {
            return Err(AccessError::PermissionDenied("cannot revoke owner".into()));
        }
        self.entries.remove(subject);
        Ok(())
    }

    /// Check if subject has permission
    pub fn check(&self, subject: &str, permission: Permission) -> bool {
        self.entries
            .get(subject)
            .map(|e| e.has_permission(permission))
            .unwrap_or(false)
    }

    /// Get entry for subject
    pub fn get_entry(&self, subject: &str) -> Option<&AccessEntry> {
        self.entries.get(subject)
    }

    /// List all subjects with access
    pub fn list_subjects(&self) -> Vec<&str> {
        self.entries.keys().map(|s| s.as_str()).collect()
    }

    /// Get owner
    pub fn owner(&self) -> Option<&str> {
        self.owner.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_permissions() {
        assert!(Role::Viewer.permissions().contains(Permission::READ));
        assert!(!Role::Viewer.permissions().contains(Permission::WRITE));

        assert!(Role::Editor.permissions().contains(Permission::READ));
        assert!(Role::Editor.permissions().contains(Permission::WRITE));

        assert!(Role::Admin.permissions().contains(Permission::ADMIN));
    }

    #[test]
    fn test_access_control() {
        let mut acl = AccessControl::with_owner("alice");

        acl.grant_role("bob", Role::Editor);
        acl.grant_role("charlie", Role::Viewer);

        assert!(acl.check("alice", Permission::ADMIN));
        assert!(acl.check("bob", Permission::WRITE));
        assert!(!acl.check("bob", Permission::ADMIN));
        assert!(acl.check("charlie", Permission::READ));
        assert!(!acl.check("charlie", Permission::WRITE));
    }

    #[test]
    fn test_grant_deny_permission() {
        let mut acl = AccessControl::new();

        // Grant read + extra write
        acl.grant_role("user1", Role::Viewer);
        acl.grant_permission("user1", Permission::WRITE);

        assert!(acl.check("user1", Permission::READ));
        assert!(acl.check("user1", Permission::WRITE));

        // Deny read
        acl.deny_permission("user1", Permission::READ);
        assert!(!acl.check("user1", Permission::READ));
        assert!(acl.check("user1", Permission::WRITE));
    }

    #[test]
    fn test_revoke() {
        let mut acl = AccessControl::with_owner("owner");
        acl.grant_role("user1", Role::Editor);

        // Can revoke user
        assert!(acl.revoke("user1").is_ok());
        assert!(!acl.check("user1", Permission::READ));

        // Cannot revoke owner
        assert!(acl.revoke("owner").is_err());
    }

    #[test]
    fn test_unknown_user() {
        let acl = AccessControl::new();
        assert!(!acl.check("unknown", Permission::READ));
    }
}
