#!/usr/bin/env perl
use strict; use warnings; use File::Basename qw(dirname); use Cwd qw(abs_path); use File::Spec;
my $root = abs_path(File::Spec->catdir(dirname(__FILE__), '..', '..'));
sub run_step { my ($name,@cmd)=@_; print "== $name ==\n"; system(@cmd)==0 or die "$name failed\n"; }
chdir $root or die "Cannot enter $root: $!";
if (-f 'web/package.json') {
  my $npm = $^O eq 'MSWin32' ? 'npm.cmd' : 'npm';
  chdir 'web' or die "Cannot enter web: $!";
  run_step('Web install', $npm, (-f 'package-lock.json' ? 'ci' : 'install'));
  run_step('Web build', $npm, 'run', 'build');
  chdir $root or die "Cannot return to root: $!";
}
run_step('Python validation', $^X, '-m', 'compileall', '-q', 'python') if -d 'python';
run_step('.NET build','dotnet','build','dotnet','-c','Release','--nologo') if -d 'dotnet';
print "Perl build orchestrator completed.\n";
