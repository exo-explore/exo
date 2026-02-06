{
  lib,
  stdenvNoCC,
  xcodePlatform,
  sdkVersion,
}:

let
  inherit (lib.generators) toPlist;

  Info = rec {
    CFBundleIdentifier = "com.apple.platform.${Name}";
    DefaultProperties = {
      COMPRESS_PNG_FILES = "NO";
      DEPLOYMENT_TARGET_SETTING_NAME = stdenvNoCC.hostPlatform.darwinMinVersionVariable;
      STRIP_PNG_TEXT = "NO";
    };
    Description = if stdenvNoCC.hostPlatform.isMacOS then "macOS" else "iOS";
    FamilyIdentifier = lib.toLower xcodePlatform;
    FamilyName = Description;
    Identifier = CFBundleIdentifier;
    MinimumSDKVersion = stdenvNoCC.hostPlatform.darwinMinVersion;
    Name = lib.toLower xcodePlatform;
    Type = "Platform";
    Version = sdkVersion;
  };

  # These files are all based off of Xcode spec files found in
  # /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/Library/Xcode/PrivatePlugIns/IDEOSXSupportCore.ideplugin/Contents/Resources.

  # Based off of the "MacOSX Architectures.xcspec" file. All i386 stuff
  # is removed because NixPkgs only supports darwin-x86_64 and darwin-arm64.
  Architectures = [
    {
      Identifier = "Standard";
      Type = "Architecture";
      Name = "Standard Architectures (Apple Silicon, 64-bit Intel)";
      RealArchitectures = [
        "arm64"
        "x86_64"
      ];
      ArchitectureSetting = "ARCHS_STANDARD";
    }
    {
      Identifier = "Universal";
      Type = "Architecture";
      Name = "Universal (Apple Silicon, 64-bit Intel)";
      RealArchitectures = [
        "arm64"
        "x86_64"
      ];
      ArchitectureSetting = "ARCHS_STANDARD_32_64_BIT";
    }
    {
      Identifier = "Native";
      Type = "Architecture";
      Name = "Native Architecture of Build Machine";
      ArchitectureSetting = "NATIVE_ARCH_ACTUAL";
    }
    {
      Identifier = "Standard64bit";
      Type = "Architecture";
      Name = "Apple Silicon, 64-bit Intel";
      RealArchitectures = [
        "arm64"
        "x86_64"
      ];
      ArchitectureSetting = "ARCHS_STANDARD_64_BIT";
    }
    {
      Identifier = stdenvNoCC.hostPlatform.darwinArch;
      Type = "Architecture";
      Name = "Apple Silicon or Intel 64-bit";
    }
    {
      Identifier = "Standard_Including_64_bit";
      Type = "Architecture";
      Name = "Standard Architectures (including 64-bit)";
      RealArchitectures = [
        "arm64"
        "x86_64"
      ];
      ArchitectureSetting = "ARCHS_STANDARD_INCLUDING_64_BIT";
    }
  ];

  # Based off of the "MacOSX Package Types.xcspec" file. Only keep the
  # bare minimum needed.
  PackageTypes = [
    {
      Identifier = "com.apple.package-type.mach-o-executable";
      Type = "PackageType";
      Name = "Mach-O Executable";
      DefaultBuildSettings = {
        EXECUTABLE_NAME = "$(EXECUTABLE_PREFIX)$(PRODUCT_NAME)$(EXECUTABLE_VARIANT_SUFFIX)$(EXECUTABLE_SUFFIX)";
        EXECUTABLE_PATH = "$(EXECUTABLE_NAME)";
      };
      ProductReference = {
        FileType = "compiled.mach-o.executable";
        Name = "$(EXECUTABLE_NAME)";
      };
    }
    {
      Identifier = "com.apple.package-type.mach-o-objfile";
      Type = "PackageType";
      Name = "Mach-O Object File";
      DefaultBuildSettings = {
        EXECUTABLE_NAME = "$(EXECUTABLE_PREFIX)$(PRODUCT_NAME)$(EXECUTABLE_VARIANT_SUFFIX)$(EXECUTABLE_SUFFIX)";
        EXECUTABLE_PATH = "$(EXECUTABLE_NAME)";
      };
      ProductReference = {
        FileType = "compiled.mach-o.objfile";
        Name = "$(EXECUTABLE_NAME)";
      };
    }
    {
      Identifier = "com.apple.package-type.mach-o-dylib";
      Type = "PackageType";
      Name = "Mach-O Dynamic Library";
      DefaultBuildSettings = {
        EXECUTABLE_NAME = "$(EXECUTABLE_PREFIX)$(PRODUCT_NAME)$(EXECUTABLE_VARIANT_SUFFIX)$(EXECUTABLE_SUFFIX)";
        EXECUTABLE_PATH = "$(EXECUTABLE_NAME)";
      };
      ProductReference = {
        FileType = "compiled.mach-o.dylib";
        Name = "$(EXECUTABLE_NAME)";
      };
    }
    {
      Identifier = "com.apple.package-type.static-library";
      Type = "PackageType";
      Name = "Mach-O Static Library";
      DefaultBuildSettings = {
        EXECUTABLE_PREFIX = "lib";
        EXECUTABLE_SUFFIX = ".a";
        EXECUTABLE_NAME = "$(EXECUTABLE_PREFIX)$(PRODUCT_NAME)$(EXECUTABLE_VARIANT_SUFFIX)$(EXECUTABLE_SUFFIX)";
        EXECUTABLE_PATH = "$(EXECUTABLE_NAME)";
      };
      ProductReference = {
        FileType = "archive.ar";
        Name = "$(EXECUTABLE_NAME)";
        IsLaunchable = "NO";
      };
    }
    {
      Identifier = "com.apple.package-type.wrapper";
      Type = "PackageType";
      Name = "Wrapper";
      DefaultBuildSettings = {
        WRAPPER_SUFFIX = ".bundle";
        WRAPPER_NAME = "$(WRAPPER_PREFIX)$(PRODUCT_NAME)$(WRAPPER_SUFFIX)";
        CONTENTS_FOLDER_PATH = "$(WRAPPER_NAME)/Contents";
        EXECUTABLE_NAME = "$(EXECUTABLE_PREFIX)$(PRODUCT_NAME)$(EXECUTABLE_VARIANT_SUFFIX)$(EXECUTABLE_SUFFIX)";
        EXECUTABLE_FOLDER_PATH = "$(CONTENTS_FOLDER_PATH)/MacOS";
        EXECUTABLE_PATH = "$(EXECUTABLE_FOLDER_PATH)/$(EXECUTABLE_NAME)";
        INFOPLIST_PATH = "$(CONTENTS_FOLDER_PATH)/Info.plist";
        INFOSTRINGS_PATH = "$(LOCALIZED_RESOURCES_FOLDER_PATH)/InfoPlist.strings";
        PKGINFO_PATH = "$(CONTENTS_FOLDER_PATH)/PkgInfo";
        PBDEVELOPMENTPLIST_PATH = "$(CONTENTS_FOLDER_PATH)/pbdevelopment.plist";
        VERSIONPLIST_PATH = "$(CONTENTS_FOLDER_PATH)/version.plist";
        PUBLIC_HEADERS_FOLDER_PATH = "$(CONTENTS_FOLDER_PATH)/Headers";
        PRIVATE_HEADERS_FOLDER_PATH = "$(CONTENTS_FOLDER_PATH)/PrivateHeaders";
        EXECUTABLES_FOLDER_PATH = "$(CONTENTS_FOLDER_PATH)/Executables";
        FRAMEWORKS_FOLDER_PATH = "$(CONTENTS_FOLDER_PATH)/Frameworks";
        SHARED_FRAMEWORKS_FOLDER_PATH = "$(CONTENTS_FOLDER_PATH)/SharedFrameworks";
        SHARED_SUPPORT_FOLDER_PATH = "$(CONTENTS_FOLDER_PATH)/SharedSupport";
        UNLOCALIZED_RESOURCES_FOLDER_PATH = "$(CONTENTS_FOLDER_PATH)/Resources";
        LOCALIZED_RESOURCES_FOLDER_PATH = "$(UNLOCALIZED_RESOURCES_FOLDER_PATH)/$(DEVELOPMENT_LANGUAGE).lproj";
        DOCUMENTATION_FOLDER_PATH = "$(LOCALIZED_RESOURCES_FOLDER_PATH)/Documentation";
        PLUGINS_FOLDER_PATH = "$(CONTENTS_FOLDER_PATH)/PlugIns";
        SCRIPTS_FOLDER_PATH = "$(UNLOCALIZED_RESOURCES_FOLDER_PATH)/Scripts";
      };
      ProductReference = {
        FileType = "wrapper.cfbundle";
        Name = "$(WRAPPER_NAME)";
        IsLaunchable = "NO";
      };
    }
    {
      Identifier = "com.apple.package-type.wrapper.application";
      Type = "PackageType";
      BasedOn = "com.apple.package-type.wrapper";
      Name = "Application Wrapper";
      DefaultBuildSettings = {
        GENERATE_PKGINFO_FILE = "YES";
      };
      ProductReference = {
        FileType = "wrapper.application";
        Name = "$(WRAPPER_NAME)";
        IsLaunchable = "YES";
      };
    }
  ];

  # Based off of the "MacOSX Product Types.xcspec" file. All
  # bundles/wrapper are removed, because we prefer dynamic products in
  # NixPkgs.
  ProductTypes = [
    {
      Identifier = "com.apple.product-type.tool";
      Type = "ProductType";
      Name = "Command-line Tool";
      PackageTypes = [ "com.apple.package-type.mach-o-executable" ];
    }
    {
      Identifier = "com.apple.product-type.objfile";
      Type = "ProductType";
      Name = "Object File";
      PackageTypes = [ "com.apple.package-type.mach-o-objfile" ];
    }
    {
      Identifier = "com.apple.product-type.library.dynamic";
      Type = "ProductType";
      Name = "Dynamic Library";
      PackageTypes = [ "com.apple.package-type.mach-o-dylib" ];
      DefaultBuildProperties = {
        FULL_PRODUCT_NAME = "$(EXECUTABLE_NAME)";
        MACH_O_TYPE = "mh_dylib";
        REZ_EXECUTABLE = "YES";
        EXECUTABLE_SUFFIX = ".$(EXECUTABLE_EXTENSION)";
        EXECUTABLE_EXTENSION = "dylib";
        DYLIB_COMPATIBILITY_VERSION = "1";
        DYLIB_CURRENT_VERSION = "1";
        FRAMEWORK_FLAG_PREFIX = "-framework";
        LIBRARY_FLAG_PREFIX = "-l";
        LIBRARY_FLAG_NOSPACE = "YES";
        STRIP_STYLE = "debugging";
        GCC_INLINES_ARE_PRIVATE_EXTERN = "YES";
        CODE_SIGNING_ALLOWED = "YES";
        CODE_SIGNING_REQUIRED = "NO";
      };
    }
    {
      Identifier = "com.apple.product-type.library.static";
      Type = "ProductType";
      Name = "Static Library";
      PackageTypes = [ "com.apple.package-type.static-library" ];
      DefaultBuildProperties = {
        FULL_PRODUCT_NAME = "$(EXECUTABLE_NAME)";
        MACH_O_TYPE = "staticlib";
        REZ_EXECUTABLE = "YES";
        EXECUTABLE_PREFIX = "lib";
        EXECUTABLE_SUFFIX = ".$(EXECUTABLE_EXTENSION)";
        EXECUTABLE_EXTENSION = "a";
        FRAMEWORK_FLAG_PREFIX = "-framework";
        LIBRARY_FLAG_PREFIX = "-l";
        LIBRARY_FLAG_NOSPACE = "YES";
        STRIP_STYLE = "debugging";
        SEPARATE_STRIP = "YES";
        CLANG_ENABLE_MODULE_DEBUGGING = "NO";
      };
    }
    {
      Type = "ProductType";
      Identifier = "com.apple.product-type.bundle";
      Name = "Bundle";
      DefaultBuildProperties = {
        FULL_PRODUCT_NAME = "$(WRAPPER_NAME)";
        MACH_O_TYPE = "mh_bundle";
        WRAPPER_PREFIX = "";
        WRAPPER_SUFFIX = ".$(WRAPPER_EXTENSION)";
        WRAPPER_EXTENSION = "bundle";
        WRAPPER_NAME = "$(WRAPPER_PREFIX)$(PRODUCT_NAME)$(WRAPPER_SUFFIX)";
        FRAMEWORK_FLAG_PREFIX = "-framework";
        LIBRARY_FLAG_PREFIX = "-l";
        LIBRARY_FLAG_NOSPACE = "YES";
        STRIP_STYLE = "non-global";
      };
      PackageTypes = [ "com.apple.package-type.wrapper" ];
      IsWrapper = "YES";
      HasInfoPlist = "YES";
      HasInfoPlistStrings = "YES";
    }
    {
      Identifier = "com.apple.product-type.application";
      Type = "ProductType";
      BasedOn = "com.apple.product-type.bundle";
      Name = "Application";
      DefaultBuildProperties = {
        MACH_O_TYPE = "mh_execute";
        WRAPPER_SUFFIX = ".$(WRAPPER_EXTENSION)";
        WRAPPER_EXTENSION = "app";
      };
      PackageTypes = [ "com.apple.package-type.wrapper.application" ];
    }
    {
      Type = "ProductType";
      Identifier = "com.apple.product-type.framework";
      Name = "Bundle";
      DefaultBuildProperties = {
        FULL_PRODUCT_NAME = "$(WRAPPER_NAME)";
        MACH_O_TYPE = "mh_bundle";
        WRAPPER_PREFIX = "";
        WRAPPER_SUFFIX = ".$(WRAPPER_EXTENSION)";
        WRAPPER_EXTENSION = "bundle";
        WRAPPER_NAME = "$(WRAPPER_PREFIX)$(PRODUCT_NAME)$(WRAPPER_SUFFIX)";
        FRAMEWORK_FLAG_PREFIX = "-framework";
        LIBRARY_FLAG_PREFIX = "-l";
        LIBRARY_FLAG_NOSPACE = "YES";
        STRIP_STYLE = "non-global";
      };
      PackageTypes = [ "com.apple.package-type.wrapper" ];
      IsWrapper = "YES";
      HasInfoPlist = "YES";
      HasInfoPlistStrings = "YES";
    }
  ];

  ToolchainInfo = {
    Identifier = "com.apple.dt.toolchain.XcodeDefault";
  };
in
{
  "Info.plist" = builtins.toFile "Info.plist" (toPlist { escape = true; } Info);
  "ToolchainInfo.plist" = builtins.toFile "ToolchainInfo.plist" (
    toPlist { escape = true; } ToolchainInfo
  );
  "Architectures.xcspec" = builtins.toFile "Architectures.xcspec" (
    toPlist { escape = true; } Architectures
  );
  "PackageTypes.xcspec" = builtins.toFile "PackageTypes.xcspec" (
    toPlist { escape = true; } PackageTypes
  );
  "ProductTypes.xcspec" = builtins.toFile "ProductTypes.xcspec" (
    toPlist { escape = true; } ProductTypes
  );
}
